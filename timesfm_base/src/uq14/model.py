"""Joint four-station TimesFM encoder with a shared 14-day output head.

Pinned upstream source: e31dadd84cb26bd5153fde6687502b8312e918fb.
The input is strictly [B,4,14]. TimesFM left-pads the 14 observed days with
18 masked values to one 32-point patch; its native cross-station attention is
retained. Station order is part of the dataset contract, normally BRW/MLO/SMO/SPO.

Every station uses the same MLP and its own native RevIN/trend coordinates.
One trajectory draws ONE 16-dimensional noise vector, broadcast to four station
heads. It is a 16-dimensional latent model, not four independent noise vectors.
The native quantile head is used for initialization and separate baselines only.
The learned head never adds native forecasts to its own predictions.

The native zero scale is deliberately preserved. A station with zero historical
RevIN scale has a degenerate forecast under this parametrization; its head cannot
learn an additive correction or nonzero conditional spread at that input.

Upstream decode disables gradients. Unwrapping its decorator is an experimental
training adapter, not an official TimesFM fine-tuning API.
"""

from __future__ import annotations

import inspect
import math
from pathlib import Path
from typing import Mapping

import torch
from torch import nn

from .coordinates import _HistoryFeaturesView, _historical_trend
from .backbone import load_backbone
from .initialization import initialize_silu_head_from_linear


class Station14ForecastModel(nn.Module):
    """Four historical station channels -> four 14-day predictive trajectories.

    ``encode`` returns features[B,4,D], center[B,4,14], scale[B,4,1].
    Frozen features may be cached; after unfreezing any block they must be
    recomputed for every optimization step. No future labels enter this API.
    """

    architecture_version = "four_station_14_to_14_shared_v1"
    station_count = 4
    context_days = 14
    horizon = 14

    def __init__(
        self,
        backbone: nn.Module,
        *,
        head_type: str = "engression",
        noise_dim: int = 16,
        unfreeze_last_n: int = 0,
        hidden_dim: int = 64,
        noise_scale: float = 0.05,
    ) -> None:
        super().__init__()
        if head_type not in {"deterministic", "engression"}:
            raise ValueError("head_type must be deterministic or engression.")
        if not isinstance(noise_dim, int) or isinstance(noise_dim, bool) or noise_dim < 1:
            raise ValueError("noise_dim must be a positive integer.")
        if not isinstance(hidden_dim, int) or isinstance(hidden_dim, bool) or hidden_dim < 2 * self.horizon:
            raise ValueError("hidden_dim must be an integer >= 28 for paired-SiLU initialization.")
        if not math.isfinite(noise_scale) or noise_scale < 0:
            raise ValueError("noise_scale must be finite and nonnegative.")
        if (backbone.input_patch_len, backbone.output_patch_len) != (32, 64):
            raise ValueError("This adapter requires the pinned input/output patch lengths 32/64.")
        if getattr(backbone, "input_transform", "identity") != "identity":
            raise ValueError("Only the pinned identity input_transform is supported.")
        if not backbone.use_variate_attention:
            raise ValueError("Native joint cross-station attention must be enabled.")
        if 0.5 not in backbone.quantiles:
            raise ValueError("The checkpoint must contain a 0.5 quantile.")
        self.backbone = backbone
        self.head_type = head_type
        self.noise_dim = noise_dim
        self.hidden_dim = hidden_dim
        self.noise_scale = float(noise_scale)
        self.median_index = list(backbone.quantiles).index(0.5)
        self.quantile_levels = tuple(float(q) for q in backbone.quantiles)
        self.feature_dim = backbone.transformer_config.transformer.model_dims
        param = next(backbone.parameters())
        self.head = nn.Sequential(
            nn.Linear(self.feature_dim + (noise_dim if head_type == "engression" else 0), hidden_dim),
            nn.SiLU(), nn.Linear(hidden_dim, hidden_dim), nn.SiLU(),
            nn.Linear(hidden_dim, self.horizon),
        ).to(device=param.device, dtype=param.dtype)
        rows = torch.arange(self.horizon, device=param.device) * backbone.num_quantiles + self.median_index
        native_weight = backbone.output_head.weight.detach()[rows]
        native_bias = backbone.output_head.bias.detach()[rows] if backbone.output_head.bias is not None else None
        initialize_silu_head_from_linear(
            self.head, native_weight, native_bias,
            noise_dim=noise_dim if head_type == "engression" else 0,
            noise_scale=noise_scale if head_type == "engression" else 0.0,
        )
        self.configure_trainable(unfreeze_last_n)
        self.backbone.eval()

    @classmethod
    def from_pretrained(
        cls, model_dir: str | Path, device: str | torch.device = "cpu", **kwargs,
    ) -> "Station14ForecastModel":
        """Load an existing pretrained directory, with no random fallback."""
        if not Path(model_dir).is_dir():
            raise ValueError("model_dir must be an existing local pretrained checkpoint directory.")
        return cls(load_backbone(model_dir, device=device), **kwargs)

    def configure_trainable(self, unfreeze_last_n: int) -> None:
        layers = self.backbone.transformer_stack.layers
        if (not isinstance(unfreeze_last_n, int) or isinstance(unfreeze_last_n, bool)
                or not 0 <= unfreeze_last_n <= len(layers)):
            raise ValueError(f"unfreeze_last_n must be an integer from 0 to {len(layers)}.")
        self.backbone.requires_grad_(False)
        if unfreeze_last_n:
            for layer in layers[-unfreeze_last_n:]:
                layer.requires_grad_(True)
        self.head.requires_grad_(True)
        self.unfreeze_last_n = unfreeze_last_n

    def train(self, mode: bool = True) -> "Station14ForecastModel":
        super().train(mode)
        self.backbone.eval()
        return self

    def trainable_summary(self) -> dict[str, int | float | str | bool]:
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {
            "architecture_version": self.architecture_version,
            "total": total, "trainable": trainable, "frozen": total - trainable,
            "backbone_trainable": sum(p.numel() for p in self.backbone.parameters() if p.requires_grad),
            "head_trainable": sum(p.numel() for p in self.head.parameters() if p.requires_grad),
            "unfreeze_last_n": self.unfreeze_last_n,
            "transformer_blocks": len(self.backbone.transformer_stack.layers),
            "station_count": self.station_count, "context_days": self.context_days,
            "horizon": self.horizon, "joint_output_dim": self.station_count * self.horizon,
            "feature_dim_per_station": self.feature_dim,
            "head_type": self.head_type,
            "noise_dim_per_joint_trajectory": self.noise_dim if self.head_type == "engression" else 0,
            "noise_shared_across_stations": self.head_type == "engression",
            "native_cross_station_attention": True,
            "shared_station_head": True, "native_zero_scale_preserved": True,
            "trainable_fraction": trainable / total,
        }

    parameter_summary = trainable_summary

    def _context(self, context: torch.Tensor) -> torch.Tensor:
        if (not isinstance(context, torch.Tensor) or context.ndim != 3 or context.shape[0] < 1
                or tuple(context.shape[1:]) != (self.station_count, self.context_days)):
            raise ValueError("context must have shape [batch,4,14] with nonempty batch.")
        if not context.is_floating_point():
            raise TypeError("context must be floating-point station observations in ppm.")
        param = next(self.backbone.parameters())
        context = context.to(device=param.device, dtype=param.dtype)
        if not bool(torch.isfinite(context).all()):
            raise ValueError("Context must be finite; handle missing historical inputs explicitly upstream.")
        return context

    def encode(self, context: torch.Tensor) -> dict[str, torch.Tensor]:
        """Read historical features and each station's own native coordinates.

        For the single historical patch p=0, output is c_i+s_i*g(h_i,epsilon).
        c_i is the native detrended running mean plus that station's extrapolated
        historical trend; s_i is its exact running standard deviation. Neither
        station averaging nor a scale floor is applied.
        """
        context = self._context(context)
        decode = inspect.unwrap(type(self.backbone).decode)
        if decode is type(self.backbone).decode:
            raise RuntimeError("Expected the pinned TimesFM3 no_grad wrapper; re-audit the upstream API.")
        _, aux = decode(
            _HistoryFeaturesView(self.backbone), target=context,
            horizon=self.horizon, return_aux_outputs=True,
        )
        patch = math.ceil(self.context_days / self.backbone.input_patch_len) - 1
        hidden = aux["__call__:transformer_output"][:, :, patch]
        mean, std = aux["revin_stats"]
        trend = _historical_trend(
            self.backbone, context.reshape(-1, self.context_days), self.horizon,
        ).reshape(context.shape[0], self.station_count, self.horizon)
        return {
            "features": hidden,
            "center": mean[:, :, patch, None] + trend,
            "scale": std[:, :, patch, None],
        }

    def sample_from_encoded(
        self, encoded: Mapping[str, torch.Tensor], n_samples: int = 1,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        """Return [B,S,4,14] ppm samples; deterministic output always has S=1.

        Exactly one [B,S,Z] Gaussian noise array is drawn and broadcast across
        the station axis. A CPU generator may be used for reproducible MPS runs.
        """
        if not isinstance(n_samples, int) or isinstance(n_samples, bool) or n_samples < 1:
            raise ValueError("n_samples must be a positive integer.")
        if set(encoded) != {"features", "center", "scale"}:
            raise ValueError("encoded must contain exactly features, center, and scale.")
        values = [encoded[key] for key in ("features", "center", "scale")]
        if not all(isinstance(value, torch.Tensor) for value in values):
            raise TypeError("Encoded values must be tensors.")
        features, center, scale = values
        if (features.ndim != 3 or features.shape[0] < 1
                or tuple(features.shape[1:]) != (self.station_count, self.feature_dim)):
            raise ValueError(f"features must have shape [batch,4,{self.feature_dim}].")
        batch = features.shape[0]
        if center.shape != (batch, self.station_count, self.horizon) or scale.shape != (batch, self.station_count, 1):
            raise ValueError("center/scale must have shapes [batch,4,14]/[batch,4,1].")
        if not all(value.is_floating_point() for value in values):
            raise TypeError("Encoded values must be floating-point tensors.")
        param = next(self.head.parameters())
        features, center, scale = [value.to(device=param.device, dtype=param.dtype) for value in values]
        if not all(bool(torch.isfinite(value).all()) for value in (features, center, scale)) or bool((scale < 0).any()):
            raise ValueError("Encoded values must be finite and native scales nonnegative.")
        if self.head_type == "deterministic":
            prediction = self.head(features)[:, None]
        else:
            expanded = features[:, None].expand(-1, n_samples, -1, -1)
            noise_device = generator.device if generator is not None else features.device
            noise = torch.randn(
                batch, n_samples, self.noise_dim,
                dtype=features.dtype, device=noise_device, generator=generator,
            ).to(features.device)
            noise = noise[:, :, None].expand(-1, -1, self.station_count, -1)
            prediction = self.head(torch.cat((expanded, noise), dim=-1))
        return center[:, None] + scale[:, None] * prediction

    def forward(
        self, context: torch.Tensor, num_samples: int = 1,
        generator: torch.Generator | None = None, *, n_samples: int | None = None,
    ) -> torch.Tensor:
        if n_samples is not None:
            if num_samples != 1 and num_samples != n_samples:
                raise ValueError("num_samples and n_samples disagree.")
            num_samples = n_samples
        return self.sample_from_encoded(self.encode(context), num_samples, generator)

    @torch.no_grad()
    def native_quantiles(self, context: torch.Tensor, *, sort_quantiles: bool = True) -> torch.Tensor:
        """Separate native [B,4,14,Q] baseline; sorting is per station and lead.

        Save this before unfreezing to retain the pretrained baseline. Native
        marginal quantiles do not specify a 56-dimensional joint distribution.
        """
        quantiles = self.backbone.decode(target=self._context(context), horizon=self.horizon)
        return quantiles.sort(dim=-1).values if sort_quantiles else quantiles

    @torch.no_grad()
    def native_raw_q50(self, context: torch.Tensor) -> torch.Tensor:
        """Raw q50 [B,4,14], before quantile sorting, for warm-start parity."""
        return self.native_quantiles(context, sort_quantiles=False)[..., self.median_index]


StationForecastModel = Station14ForecastModel

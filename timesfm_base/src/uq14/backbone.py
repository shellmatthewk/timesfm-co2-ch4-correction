"""Load pinned pretrained TimesFM without random fallback."""
from pathlib import Path
from typing import Any
import torch
from torch import nn

def load_backbone(
    checkpoint_path_or_id: str | Path = "google/timesfm-3.0-pytorch",
    revision: str | None = None,
    device: str | torch.device = "cpu",
) -> nn.Module:
    """Load real pretrained TimesFM3 weights from a Hub ID or saved directory.

    Checkpoint directories must contain Hugging Face ``config.json`` and model
    weights. Loading is performed on CPU first to avoid duplicate GPU allocation.
    No random-initialization fallback is allowed when a download/load fails.
    """
    from timesfm3.torch.model import TimesFM3Torch

    path = Path(checkpoint_path_or_id)
    if path.is_file():
        raise ValueError("Pass a pretrained directory containing config.json, not a weight file.")
    kwargs: dict[str, Any] = {"map_location": "cpu"}
    if revision is not None:
        kwargs["revision"] = revision
    model = TimesFM3Torch.from_pretrained(str(checkpoint_path_or_id), **kwargs)
    if getattr(model, "input_transform", "identity") != "identity":
        raise ValueError("This adapter currently supports identity input_transform only.")
    model.to(device=device, dtype=torch.float32)
    model.eval()
    return model

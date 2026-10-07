# TimesFM + CH₄ 修正：BRW、MLO 两站 CO₂ 的 14 天概率预测

**New: configurable solar time-series experiments.** See [SOLAR_README.md](SOLAR_README.md) for SunPy GOES XRS ingestion and the numerical forecasting workflow. History length, forecast horizon, channels, data periods and training amounts can be changed in one JSON file. The original CO₂ scripts and saved results are preserved.

**English summary.** Probabilistic 14-day forecasts of daily CO₂ at two NOAA stations, Utqiaġvik (BRW, Alaska) and Mauna Loa (MLO, Hawaii), built on the frozen TimesFM 3 foundation model. Ten schemes are compared on a rolling large-sample test (test years 2020–2025, confirmation 2014–2019):
- TimesFM itself;
- a basic Engression head;
- a median-locked residual Engression;
- two schemes whose interval width follows a learned rule of the situation (T1, N1);
- TimesFM with CH₄ as a covariate;
- four schemes corrected by a bidirectional LSTM that reads the same-station CH₄ observed during the 14 forecast days.

The best schemes, N1 + LSTM and T1 + LSTM, lower the 80% interval score (MIS80) by 18–24% relative to TimesFM with CH₄ covariates, at both stations in both periods. Schemes 6–10 use CH₄ observed during the forecast window, so they are conditional predictions (for example for filling gaps in CO₂ records), not pure forecasts.

`python make_table.py` rebuilds the results table from the per-window scores in `results/` (numpy only). The code in `src/` reproduces everything from the raw NOAA files. The TimesFM 3 weights must be downloaded separately; they have a non-commercial license.

---

## 这个项目是什么

- **任务**：用过去 14 天的每日观测，预测 BRW、MLO 两个站未来 14 天每天的 CO₂（单位 ppm），并给出不确定性。每个方法都输出一组预测样本或分位数，80% 区间等都由它们得到。
- **底座**：冻结的 TimesFM 3，不改它的参数，只读这两个站的历史。
- **比较**：十个方案放在同一张表里，用大样本滚动检验来比。

## 十个方案

| 编号 | 方案 | 怎么做 | 用到未来 14 天的 CH₄ | 代码 |
|---:|---|---|:---:|---|
| 1 | TimesFM 不修正 | TimesFM 3 读两站过去 14 天的 CO₂，用它自己给的分位数抽样 | 否 | `src/rolling_residual.py`（`fold tfm`）、`src/timesfm_rnet.py`（抽样） |
| 2 | 基础 Engression（1280 维噪声） | TimesFM 冻结，后面接 Engression 头：特征和 1280 维随机噪声进去，样本出来；不锁中位数，两站共用一份噪声，用 Energy Score 训练 | 否 | `src/rolling_basic.py` |
| 3 | 锁中位数 Engression | 残差 Engression：样本 = TimesFM 中位数 + 尺度 ×（网络输出 − 样本中位数），所以样本的中位数锁在 TimesFM 的中位数上；每站一份噪声 | 否 | `src/rolling_residual.py`（`train co2`） |
| 4 | T1：TimesFM + 随情况变的宽度 | TimesFM 自己的分布，围绕中位数乘一个放大倍数。倍数由小网络根据季节、TimesFM 的区间宽度、最近的数值尺度、过去 14 天的日波动和站点学出来 | 否 | `src/rolling_noise.py`，模型在 `src/noise_study.py` |
| 5 | N1：Engression + 噪声大小随情况变 | 锁中位数 Engression，但噪声先乘同样方式学出来的倍数，再送进网络 | 否 | 同上 |
| 6 | TimesFM + CH₄ 协变量 | 把同站 CH₄ 过去 14 天和未来 14 天的实测值作为协变量交给 TimesFM 3，不训练 | 是 | `multigas/src/mg/timesfm_covariates.py`、`src/timesfm_ch4_windows.py` |
| 7 | TimesFM + 双向 LSTM | 方案 1，再用双向 LSTM 修正：它读同站 CH₄ 未来 14 天的实测值落在预测分布的什么位置，把 CO₂ 的分布挪动、收窄 | 是 | `src/rolling_residual.py`（`fold tfm`）、`src/bilstm_correct.py` |
| 8 | 锁中位数 Engression + 双向 LSTM | 方案 3 加同样的修正（CH₄ 的位置用 CH₄ 的残差 Engression 算） | 是 | `src/rolling_residual.py`（`fold res`） |
| 9 | T1 + 双向 LSTM | 方案 4 加同样的修正（CH₄ 的位置用 TimesFM 的 CH₄ 分布算） | 是 | `src/rolling_noise.py` |
| 10 | N1 + 双向 LSTM | 方案 5 加同样的修正 | 是 | `src/rolling_noise.py` |

## 主要结果（MIS80，越低越好）

只算两种气体当天都有观测的格子，单位 ppm，三个种子平均。`*` 表示比 TimesFM + CH₄ 显著更好，`†` 表示显著更差。

| 编号 | 方案 | BRW 2020–2025 | MLO 2020–2025 | BRW 2014–2019 | MLO 2014–2019 |
|---:|---|---:|---:|---:|---:|
| 1 | TimesFM 不修正 | 9.140 | 4.047 † | 7.670 † | 4.023 † |
| 2 | 基础 Engression（1280 维噪声） | 8.692 | 4.088 † | 7.411 | 3.998 † |
| 3 | 锁中位数 Engression | 9.064 | 4.236 † | 7.536 | 4.152 † |
| 4 | T1：TimesFM + 随情况变的宽度 | 7.894 * | 3.656 | 6.906 * | 3.615 |
| 5 | N1：Engression + 噪声大小随情况变 | 7.492 * | 3.672 | 6.706 * | 3.570 |
| 6 | TimesFM + CH₄ 协变量 | 8.949 | 3.544 | 7.400 | 3.472 |
| 7 | TimesFM + 双向 LSTM | 8.275 * | 3.136 * | 7.058 | 3.112 * |
| 8 | 锁中位数 Engression + 双向 LSTM | 8.243 * | 3.297 * | 7.082 | 3.371 |
| 9 | T1 + 双向 LSTM | 6.922 * | 2.828 * | 6.061 * | 2.803 * |
| 10 | N1 + 双向 LSTM | 6.776 * | 2.905 * | 5.953 * | 2.858 * |

完整的表在 `总表.md`（`总表.json` 有更多位小数）。表里还有：每个方案减 TimesFM、减 TimesFM + CH₄ 的差值和 95% 区间，覆盖率、区间宽度、CRPS、MAE，以及当天没测到 CH₄ 的格子。两两比较（例如 N1 + LSTM 对 T1 + LSTM）在 `results/reports/large_noise*.md`。

1. **N1 + 双向 LSTM 和 T1 + 双向 LSTM 最好**：
   - 两个时期、两个站都显著好于 TimesFM + CH₄，MIS80 低 18%–24%；
   - 也显著好于 TimesFM + 双向 LSTM；
   - 两者之间没有显著差别。好处来自"按情况调整区间宽度"加上双向 LSTM，不是来自 Engression 本身。
2. **不用 CH₄ 的方案里，N1 和 T1 最好**：
   - 都显著好于 TimesFM；
   - BRW 上两期都显著好于 TimesFM + CH₄，MLO 上和它持平。
   - 学到的规则是：TimesFM 给的区间偏窄时放宽，BRW 夏季放宽最多（约 1.5–1.9 倍）。
3. **基础 Engression**：
   - BRW 上显著好于 TimesFM，MLO 上持平；
   - 它不锁中位数，点预测更准（BRW 的 MAE 1.53 对 1.67）；
   - 但区间偏窄，覆盖率只有 68%–72%。
4. **锁中位数 Engression 不修正时和 TimesFM 差不多**，MLO 上甚至更差。

## 怎么检验

- **滚动检验**：每个测试年轮流测试，前一年做验证（选轮次、提前停止），训练用到测试年的前两年。
  - 2020–2025：训练从 2010 年开始。
  - 2014–2019：训练从 2000 年开始，作为确认。
- **训练**：要训练的方案（2–5，以及 7–10 里的 LSTM）每个测试年都重新训练，种子 17、29、43，分数取平均。
- **打分**：每天一个窗口，看未来 14 天。
  - 只算 CO₂ 和 CH₄ 当天都有观测的格子，这样和 TimesFM + CH₄ 比较公平；
  - 当天没测到 CH₄ 的格子另列。
- **指标**：
  - MIS80 是 80% 区间的分数，等于区间宽度加上没包住实测值时的罚分（差多少 × 10）；
  - 覆盖率越接近 80% 越好；
  - CRPS 评整个预测分布，MAE 评中位数。
- **显著性**：差值的 95% 区间来自按月分块重抽测试年的窗口（4000 次）。
- **N1、T1 的来历**：它们是在一次小样本预试里选出来的。预试试了五种"用学到的规律拼接 Engression 噪声"的方法，训练 2000–2008 年、选轮次 2009 年、评估 2010–2013 年，代码是 `src/noise_study.py`，结果在 `results/reports/noise_pilot_*.md`。
  - T1 是看过 N1 第一个种子的预试结果后才追加的对照。
  - N1、T1 在大样本上的设置，都在看到大样本结果之前定好。

## 仓库里有什么

```
make_table.py, 总表.md, 总表.json   十个方案的总表和汇总脚本（只需要 numpy）
results/                            十个方案在 12 个测试年上的逐窗口分数，以及详细报告（results/reports/）
src/                                各方案的代码和运行脚本
multigas/src/                       数据、窗口、编码、打分用的包（mg、rq、cop、uq14）
timesfm_base/                       加载 TimesFM 的代码（src/uq14）、TimesFM 源码（vendor/，Apache-2.0）和权重锁文件
residual_engression/configs/        Engression 的设置
data/raw/                           NOAA GML 的逐日观测：CO₂（BRW、MLO、SMO、SPO），CH₄（BRW、MLO、SMO）
```

运行时生成的文件（数据、编码、训练好的模型、日志）都不放进仓库，见 `.gitignore`。

## 只重算总表

```bash
pip install numpy
python make_table.py
```

生成的 `总表.md`、`总表.json` 和仓库里的一样。

## 从头重跑

在 Apple 芯片的 Mac 上验证过。

**1. 环境**

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

**2. 下载 TimesFM 3 权重**（1.3 GB，许可为 timesfm-non-commercial-license-v1.0，只能非商业使用）。代码会按 `timesfm_base/model.lock.json` 核对文件。

```bash
python -c "from huggingface_hub import snapshot_download; snapshot_download('google/timesfm-3.0-pytorch', revision='43046b85ec22d584a13f8098c2ed39c889e129c2', local_dir='timesfm_base/models/timesfm3')"
python src/extract_native_head.py          # 从权重里取出 TimesFM 的输出层，Engression 从它开始
```

**3. 数据**：从原始文件生成窗口、TimesFM 编码和 `multigas/02_数据/multigas.npz`，约 2 分钟。

```bash
python src/prepare_data.py
```

**4. TimesFM + CH₄ 协变量（方案 6）**：约 3 分钟。

```bash
cd multigas
PYTHONPATH=src python -m mg.timesfm_covariates rolling    # 2020–2025
PYTHONPATH=src python -m mg.timesfm_covariates confirm    # 2014–2019
cd ..
```

**5. 训练和打分**：时间是在一台 15 核 Mac 上测的。

```bash
python src/run_large.py            # 方案 1、3、7、8：72 次残差 Engression 训练和每个测试年的修正，约 45 分钟
python src/run_large_noise.py      # 方案 4、5、9、10：72 次训练，约 1 小时
python src/run_large_basic.py      # 方案 2：36 次训练，约 10 分钟
python src/timesfm_ch4_windows.py  # 方案 6 的逐窗口分数
python make_table.py
```

**复现情况**：我们在 Apple 芯片的 Mac 上，用这个仓库从原始数据重跑并逐位比对。
- 第 3 步生成的数据文件和原来的逐位相同。
- 第 4 步的 TimesFM + CH₄ 和原来的逐位相同。
- 2023 年、种子 17 的方案 2、4、5、锁中位数 Engression 的训练，以及方案 7 的修正，结果和 `results/` 里的逐位相同。用运行脚本里设定的线程数（`OMP_NUM_THREADS` 等）时才逐位相同，线程数不同时会有约 10⁻⁵ 的差别。
- 在 CPU 或 CUDA 上做 TimesFM 编码时，少数窗口可能有小差别。

## 注意

- **方案 6–10 用到了同站 CH₄ 在未来 14 天的实测值**，不是纯预测，适合补 CO₂ 的缺测、做数据质控。方案 1–5 只用过去 14 天的 CO₂。
- **"随情况变的宽度"**（方案 4、5、9、10）是用训练年份、跟着评分规则一起学出来的，没有单独的校准集，不是共形校准。但它的作用接近"按情况重新调宽度"。
- **LSTM 的训练方式略有不同**：方案 7、8 每个测试年只训练一个 LSTM（用种子 17 的样本），再用在三个种子上；方案 9、10 每个测试年、每个种子各训练一个。
- **只有两个站、一对气体。** 用南极 SPO 的 CO₂ 代替 CH₄ 做修正的小试验里，效果差很多，所以修正需要同站、同时测量的另一种气体。
- **`src/residual_rnet.py`、`src/bilstm_correct.py`、`src/timesfm_rnet.py` 里还有 14 个测试窗口的函数**，它们要用到没有放进来的早期项目文件，在这个仓库里跑不了；大样本用到的部分都能运行。

## 数据和引用

NOAA GML 的现场连续观测逐日数据，原样放在 `data/raw/`。数据可以自由使用，使用时请按每个文件开头的 `dataset_citation` 引用：

- CO₂：Pétron, G. et al. (2026), *Atmospheric Carbon Dioxide Dry Air Mole Fractions from continuous measurements at Mauna Loa, Hawaii, Utqiaġvik, Alaska, American Samoa and South Pole, 1973-present*, Version 2026-08-11, NOAA GML, https://doi.org/10.15138/yaf1-bk21
- CH₄：Pétron, G. et al. (2026), *Atmospheric methane from quasi-continuous measurements at Utqiaġvik, Alaska, Mauna Loa, Hawaii, and American Samoa, 1986-present*, Version 2026-08-11, NOAA GML, https://doi.org/10.15138/ve0c-be70

用到的方法：

- TimesFM：Das, A., Kong, W., Sen, R., Zhou, Y. (2024), *A decoder-only foundation model for time-series forecasting*, ICML 2024；代码见 https://github.com/google-research/timesfm
- Engression：Shen, X., Meinshausen, N., *Engression: Extrapolation through the lens of distributional regression*

## 许可

- **本仓库的代码和结果**：MIT 许可，见 `LICENSE`。
- **`timesfm_base/vendor/timesfm3/`**：TimesFM 的源码，Apache-2.0 许可，见 `timesfm_base/vendor/TimesFM_LICENSE`。
- **TimesFM 3 权重**：不在仓库里，许可是 timesfm-non-commercial-license-v1.0。下载和使用时请遵守，包括由它取出的 `native_output_head.npz`。
- **`data/raw/`**：NOAA GML 的公开数据，按上面的方式引用。

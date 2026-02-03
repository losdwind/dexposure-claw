# DeXposure-FM

DeXposure-FM 是一个基于 GraphPFN 的图-表 foundation model，用于**预测 DeFi 协议信用敞口网络**，支持：

- **边存在预测**（link prediction）
- **边权重预测**（exposure size）
- **节点 TVL 变化预测**
- **宏观审慎工具**（SIS/sector spillover/contagion）

> 目标是提供一个“可复现、可扩展、可直接使用”的研究与开源工具仓库。

---

## 目录

- [快速开始](#快速开始)
- [环境与硬件](#环境与硬件)
- [数据集](#数据集)
- [模型权重](#模型权重)
- [训练 / 实验 / 工具](#训练--实验--工具)
- [参数列表](#参数列表)
- [结果与输出](#结果与输出)
- [复现实验](#复现实验)
- [GraphPFN 基座模型](#graphpfn-基座模型)
- [常见问题](#常见问题)
- [许可证与致谢](#许可证与致谢)

---

## 快速开始

**1) 安装依赖（uv）**

```
uv sync
```

**2) 数据集下载（Git LFS 或脚本）**

```
git lfs install
git lfs pull
# 或者
uv run python bin/download_dataset.py
```

**3) 运行一个最小示例（CPU 友好）**

```
uv run python run_full_experiment.py --mode stats
```

**4) 宏观审慎工具（Observed 模式）**

```
uv run python run_macroprudential_tools.py observed \
  --date 2025-06-30 \
  --data-path data/historical-network_week_2025-07-01.json \
  --contagion \
  --output-dir output/macro-tools
```

你也可以用 `Makefile` 快捷命令（`make help`）。

---

## 环境与硬件

**推荐环境**

- Python **3.12.9**
- CUDA **12.1**
- PyTorch **2.2.1+cu121**
- DGL **2.1.0+cu121**

**推荐硬件**

- **H100**（完整训练 / 论文复现实验）

**CPU 可运行范围**

- `run_full_experiment.py --mode stats`
- `run_macroprudential_tools.py observed`
- `run_task2_model_based.py --plot-only`

---

## 数据集

数据集下载脚本：`bin/download_dataset.py`  
默认会下载到 `data/` 目录。

主要文件：

- `data/historical-network_week_2020-03-30.json`（约 1.1GB）
- `data/historical-network_week_2025-07-01.json`（约 76MB）
- `data/meta_df.csv`
- `data/mapping/*.json`
- `data/network_data/*.csv`

---

## 模型权重

权重已公开发布，详细说明与模型卡见：

- `checkpoints/dexposure-fm-release/README.md`

下载方式（HuggingFace）示例：

```
uv run huggingface-cli download EVIEHub/DeXposure-FM --local-dir checkpoints/
```

已发布的权重包括：

- `dexposure-fm-h1.pt`
- `dexposure-fm-h4.pt`
- `dexposure-fm-h8-h12.pt`
- `graphpfn-frozen-all-horizons.pt`
- 以及 GraphPFN / LimiX 基座权重

---

## 训练 / 实验 / 工具

### 1) Task I：多步预测（run_full_experiment.py）

**最小统计实验（CPU）**

```
uv run python run_full_experiment.py --mode stats
```

**完整比较实验（GPU）**

```
uv run python run_full_experiment.py --mode compare
```

### 2) Task II：前瞻性风险分析（run_task2_model_based.py）

```
uv run python run_task2_model_based.py --experiment all
```

### 3) 宏观审慎工具（run_macroprudential_tools.py）

**Observed**

```
uv run python run_macroprudential_tools.py observed --date 2025-06-30 \
  --data-path data/historical-network_week_2025-07-01.json
```

**Predict**

```
uv run python run_macroprudential_tools.py predict --date 2025-06-30 \
  --horizon 4 --device cuda \
  --data-path data/historical-network_week_2025-07-01.json
```

---

## 参数列表

### `run_full_experiment.py`

| 参数 | 默认值 | 说明 |
|---|---|---|
| `--mode` | `all` | all / frozen / dexposure-fm / roland / persistence / stats / compare |
| `--output-dir` | `None` | 输出目录 |
| `--epochs` | `20` | 训练轮数 |
| `--patience` | `3` | Early stopping patience |
| `--early-stop-metric` | `auprc` | 早停指标（auprc/auroc） |
| `--val-eval-every` | `1` | 每 N 轮评估验证集 |
| `--seed` | `42` | 随机种子 |
| `--holdout-start` | `2025-01-01` | 测试集起始日期 |
| `--min-train-weeks` | `104` | 最小训练周数 |
| `--val-weeks` | `24` | 验证窗口 |
| `--test-weeks` | `8` | 测试窗口 |
| `--step-weeks` | `8` | 滚动步长 |
| `--rolling` | `False` | 是否走 walk-forward |
| `--save-predictions` | `False` | 保存预测 CSV |
| `--horizons` | `1,4,8,12` | 预测步长 |
| `--gradient-clip-norm` | `1.0` | 梯度裁剪 |
| `--verbose` | `False` | Debug 日志 |

### `run_task2_model_based.py`

| 参数 | 默认值 | 说明 |
|---|---|---|
| `--experiment` | `all` | all / forward_risk / predictive_contagion / early_warning / sis_sensitivity |
| `--epochs` | `20` | 训练轮数 |
| `--seed` | `42` | 随机种子 |
| `--device` | `None` | 设备（默认自动推断） |
| `--force-retrain` | `False` | 强制重训 |
| `--frozen` | `False` | GraphPFN Frozen |
| `--output-dir` | `output/task2_model_based` | 输出目录 |
| `--quick` | `False` | 快速 smoke-test |
| `--forward-horizons` | `1,4,8,12` | forward_risk 的 horizons |
| `--contagion-horizons` | `1,4,8,12` | predictive_contagion 的 horizons |
| `--shared-model-h1` | `False` | 仅训练 h1 并复用 |
| `--max-train-pairs` | `0` | 训练 pairs 限制 |
| `--max-forward-pairs` | `0` | forward_risk 测试 pairs 限制 |
| `--max-contagion-samples` | `10` | predictive_contagion samples |
| `--plot-only` | `False` | 仅画图 |
| `--no-plot` | `False` | 跳过绘图 |
| `--reuse-results` | `False` | 复用已有结果 |

### `run_macroprudential_tools.py`

**通用参数**

| 参数 | 默认值 | 说明 |
|---|---|---|
| `--data-path` | `ExperimentConfig.data_path` | 网络快照路径 |
| `--meta-path` | `ExperimentConfig.meta_path` | 元信息路径 |
| `--top-k` | `20` | SIS Top-K |
| `--full-sis` | `False` | 输出完整 SIS |
| `--spillover-matrix` | `False` | 输出完整 spillover |
| `--contagion` | `False` | contagion 场景 |
| `--output` | `""` | 输出文件 |
| `--output-dir` | `output/macroprudential_tools` | 输出目录 |

**Observed 子命令**

| 参数 | 必填 | 说明 |
|---|---|---|
| `--date` | 是 | 日期（YYYY-MM-DD） |

**Predict 子命令**

| 参数 | 默认值 | 说明 |
|---|---|---|
| `--date` | 必填 | anchor date |
| `--horizon` | 必填 | 预测周数 |
| `--edge-threshold` | `0.5` | 边存在阈值 |
| `--epochs` | `20` | 训练轮数 |
| `--seed` | `42` | 随机种子 |
| `--device` | `None` | cpu/cuda |
| `--frozen` | `False` | 冻结 encoder |
| `--force-retrain` | `False` | 强制重训 |
| `--model-cache-dir` | `output/model_cache` | 模型缓存 |
| `--cache-tag` | `None` | 缓存 tag |
| `--train-cutoff` | `""` | 训练时间截断 |
| `--max-train-pairs` | `0` | 训练 pairs 上限 |

---

## 结果与输出

**Task I**

- `output/YYYY-MM-DD_HHMMSS/`
  - `frozen/`、`finetuned/`、`roland/`
  - `report.json`、`predictions/*.csv`、`*.log`

**Task II**

- `output/task2_model_based/`
  - `exp1_forward_risk.json`
  - `exp2_predictive_contagion.json`
  - `exp3_early_warning.json`
  - `fig_*.pdf`

**宏观审慎工具**

- `output/macroprudential_tools/observed_*.json`
- `output/macroprudential_tools/predict_*.json`

---

## 复现实验

论文复现与完整命令说明见：

- `docs/dexposure_fm_experiments.md`

---

## GraphPFN 基座模型

如果需要复现 GraphPFN 原始实验：

```
uv run bin/go.py exp/graphpfn-eval/finetune/raw/tolokers-2/tuning.toml --force
```

GraphPFN 预训练需生成图数据，详见：

- `bin/prior/README.md`

---

## 常见问题

**Q: DGL CUDA 不可用？**  
A: 检查 CUDA 版本与 DGL wheel 是否匹配（默认 cu121）。

**Q: OOM 或显存不足？**  
A: 使用 `--quick`、`--max-train-pairs` 或 `--frozen`，减少训练规模。

**Q: 加载大 JSON 很慢？**  
A: 已启用 `ijson` 流式解析（见 `run_full_experiment.py`），确保依赖已安装。

---

## 许可证与致谢

本项目采用 Apache-2.0。  
第三方组件：TabICL、LimiX（详见 `NOTICE` 和 `LICENSES/`）。


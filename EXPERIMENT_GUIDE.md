# DeXposure-FM 实验指南

> 最后更新: 2026-01-12

## 项目目标

为 NBER "AI and Economic Measurement" 会议准备论文实验，验证 GraphPFN 在 DeFi 协议间信用暴露预测任务上的表现。

### 论文核心评估任务

| 任务 | 描述 | 状态 |
|------|------|------|
| Task I | Multi-step Forecasting (h=1,3,7) | ✅ 已实现 |
| Task II | Shock Analysis (Terra/FTX) | ✅ 已实现 |
| Task III | Imputation (缺失值填补) | ✅ 已实现 |

---

## 环境准备

### 1. 进入项目目录

```bash
cd /home/figurich/CodeProjects/graph-dexposure
```

### 2. 确认依赖

```bash
# 核心依赖
pip install torch dgl torch-geometric pandas numpy scikit-learn networkx

# 可选: 大文件加载加速
pip install ijson
```

### 3. 确认数据文件

```bash
# 完整历史网络数据 (约1.1GB, 283周, 主实验用)
ls -lh data/historical-network_week_2020-03-30.json

# 最近快照数据 (约76MB, 8周, 快速测试用)
ls -lh data/historical-network_week_2025-07-01.json

# 元数据
ls -lh data/meta_df.csv

# GraphPFN 预训练checkpoint
ls -lh checkpoints/graphpfn-v1.ckpt
```

> **注意**: 主实验应使用 `historical-network_week_2020-03-30.json` 完整历史文件。

---

## 运行实验

### 快速开始 - 运行所有实验

```bash
python run_full_experiment.py --mode all --epochs 5
```

这会依次执行:
1. GraphPFN (Frozen encoder)
2. GraphPFN (Finetuned)
3. ROLAND baseline
4. Network statistics
5. Shock analysis

结果保存到 `output/full_experiment/experiment_results.json`

### 分模块运行

```bash
# 1. GraphPFN Frozen (线性探针)
python run_full_experiment.py --mode frozen --epochs 5

# 2. GraphPFN Finetuned (端到端微调)
python run_full_experiment.py --mode finetuned --epochs 5

# 3. ROLAND Baseline
python run_full_experiment.py --mode roland --epochs 5

# 4. 网络统计量
python run_full_experiment.py --mode stats

# 5. Shock Analysis (Terra/FTX事件)
python run_full_experiment.py --mode shock

# 6. 使用ROLAND进行Shock Analysis
python run_full_experiment.py --mode shock --shock-model roland

# 7. Imputation实验 (Task III)
python run_full_experiment.py --mode impute
```

### 常用参数

**实验设置:**

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--mode` | all | 实验模式 (frozen/finetuned/roland/stats/shock/impute/all) |
| `--epochs` | 5 | 训练轮数 |
| `--seed` | 42 | 随机种子 |
| `--output-dir` | output/full_experiment | 输出目录 |
| `--shock-model` | graphpfn | Shock分析使用的模型 |
| `--save-predictions` | False | 是否保存预测到 CSV |

**时序划分 (Expanding Window):**

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--holdout-start` | 2025-01-01 | Hold-out 测试集起始日期 |
| `--min-train-weeks` | 104 | 最少训练周数 (2年) |
| `--val-weeks` | 12 | 验证窗口大小 |
| `--test-weeks` | 8 | 每fold测试窗口大小 |
| `--step-weeks` | 8 | 滚动步长 |
| `--rolling` | False | 运行完整 rolling window 评估 |

### 时序划分策略 (Expanding Window Walk-Forward)

```
Timeline:
════════════════════════════════════════════════════════════════════════════

Fold 1:  [══ Train (104w) ══][Val 12w][Test 8w]
Fold 2:  [════ Train (112w) ════][Val 12w][Test 8w]
Fold 3:  [══════ Train (120w) ══════][Val 12w][Test 8w]
  ...         (训练窗口逐步扩展)
Fold N:  [══════════════ Train ══════════════][Val 12w][Test 8w]

════════════════════════════════════════════════════════════════════════════
Hold-out: [════════════ All pre-2025 Train ════════════][Val 12w][ 2025 Test ]
                                                                  (NEVER seen)
```

**为什么用 Expanding Window?**
1. **金融惯例**: 避免 look-ahead bias，符合实际交易场景
2. **多次评估**: 不依赖单一测试集，结果更稳健
3. **训练数据增长**: 随时间积累更多数据，模拟真实部署
4. **Hold-out 验证**: 2025 数据从未用于任何训练/调参

---

## 预期输出

### 1. 终端输出示例

```
============================================================
COMPARISON TABLE - Multi-step Forecasting Results
============================================================
Model                     h=1 AUPRC   h=3 AUPRC   h=7 AUPRC   Weight MAE   Recall@100
-----------------------------------------------------------------------------------------
GraphPFN (Frozen)         0.7832      0.7456      0.7123      1.82         0.8234
GraphPFN (Finetuned)      0.8142      0.7821      0.7534      1.54         0.8567
ROLAND                    0.7234      0.6987      0.6654      N/A          0.7123

============================================================
SHOCK ANALYSIS SUMMARY
============================================================
Event                     TVL Change      Edge Change     Gini Δ       HHI Δ
--------------------------------------------------------------------------------
Terra/Luna Collapse       -45.2%          -12.3%          0.0234       0.0156
FTX Collapse              -28.7%          -8.5%           0.0189       0.0098

============================================================
IMPUTATION RESULTS SUMMARY
============================================================
Mask %       Edge Recall     Edge MAE        Node MAE
------------------------------------------------------------
10           0.8923          1.2345          0.9876
20           0.8456          1.4567          1.1234
30           0.7891          1.6789          1.2567
```

### 2. 输出文件

```
output/full_experiment/
├── experiment_results.json       # 完整实验结果
├── data_quality.json             # 数据质量统计
├── network_statistics.csv        # 网络统计时序数据
├── predictions_edges_test.csv    # 边级预测 (需 --save-predictions)
└── predictions_nodes_test.csv    # 节点级预测 (需 --save-predictions)
```

### 3. 关键指标解读

**Link Prediction (边存在性):**
- AUPRC > 0.8 表示良好的预测能力 (主指标，处理类别不平衡)
- AUROC > 0.9 表示优秀的区分能力
- Recall@K: 前K个预测中正例的召回率 (K=100, 500, 1000)

**Edge Weight (边权重):**
- MAE: 平均绝对误差 (log-scale)
- Weighted MAE: 按真实边权加权的MAE (对大额暴露更敏感)

**Network Statistics:**
- Gini: 0=完全平等, 1=完全不平等
- HHI: 0=分散, 1=集中
- Density: 边数/最大可能边数
- Top-10% Concentration: 前10%节点的TVL占比

---

## 之前的实验结果

使用 `run_dexposure_experiment.py` 的单次运行结果:

```
AUPRC: 0.814
AUROC: 0.958
Weight MAE: 1.627
```

---

## 文件结构

```
graph-dexposure/
├── run_full_experiment.py      # 主实验脚本 (统一入口)
├── run_optuna_search.py        # Optuna 超参数优化
├── run_final_evaluation.py     # 最终评估 (2025 hold-out test)
├── run_dexposure_experiment.py # 单次实验脚本 (已验证可用)
├── EXPERIMENT_PLAN.md          # 详细实验计划
├── EXPERIMENT_GUIDE.md         # 本文件
├── lib/
│   └── graphpfn/               # GraphPFN 模型库
│       └── model.py
├── src/
│   └── network_statistics.py   # 网络统计量模块
├── data/
│   ├── historical-network_week_2020-03-30.json  # 完整历史 (1.1GB)
│   ├── historical-network_week_2025-07-01.json  # 最近快照 (76MB)
│   └── meta_df.csv
├── checkpoints/
│   └── graphpfn-v1.ckpt
└── output/
    ├── full_experiment/        # 完整实验结果
    │   ├── experiment_results.json
    │   ├── data_quality.json
    │   ├── network_statistics.csv
    │   ├── predictions_edges_test.csv
    │   └── predictions_nodes_test.csv
    ├── optuna/                 # 超参数优化结果
    └── final_evaluation/       # 最终评估结果
```

---

## 下一步计划

### 完整实验流程

```bash
# Step 1: 超参数优化 (可选,约6小时)
python run_optuna_search.py --n-trials 50 --timeout 21600

# Step 2: 运行完整实验 (使用优化后的超参数)
python run_full_experiment.py --mode all --epochs 10

# Step 3: 最终评估 (在2025 hold-out测试集上,多种子运行)
python run_final_evaluation.py --seeds 42 123 456 789 2024 --epochs 10

# 如果有Optuna结果,使用最佳超参数:
python run_final_evaluation.py --config output/optuna/best_params.json
```

### 待完成任务

1. **超参数优化** (可选)
   ```bash
   python run_optuna_search.py --n-trials 50
   ```

2. **最终评估** (严格时序划分: 2020-2024训练, 2025测试)
   ```bash
   python run_final_evaluation.py --epochs 10
   ```

3. **生成论文图表**
   - Multi-step forecasting 对比图
   - Shock analysis 时序变化图
   - Network statistics 演变图
   - Imputation性能曲线

4. **统计显著性测试** (已集成在 run_final_evaluation.py)
   - 多种子运行 (seeds: 42, 123, 456, 789, 2024)
   - 计算均值和标准差
   - Paired t-test + Wilcoxon test
   - 自动生成 LaTeX 表格

### 论文实验对照表

| 论文Section | 实验内容 | 脚本/函数 | 主要指标 |
|-------------|----------|-----------|----------|
| 5.2 Multi-step | h=1,3,7预测 | `run_graphpfn_experiment()` | AUPRC, AUROC, Recall@K |
| 5.2 Baselines | Frozen vs Finetuned vs ROLAND | `--mode frozen/finetuned/roland` | 对比表格 |
| 5.3 Shock Analysis | Terra/FTX事件 | `run_shock_analysis()` | TVL变化, Gini Δ, 性能退化 |
| 5.4 Network Stats | Gini/HHI/Density | `run_network_statistics()` | 时序统计 |
| 5.5 Imputation | 缺失值重建 | `run_imputation_experiment()` | Edge Recall, MAE |
| Appendix | Rolling evaluation | `expanding_window_split()` | 多fold聚合 |

---

## 常见问题

### Q: CUDA out of memory

降低batch size:
```python
# 在 ExperimentConfig 中修改
edge_batch_size: int = 10000  # 从20000降低
```

### Q: 数据加载很慢

安装 ijson 加速:
```bash
pip install ijson
```

### Q: GraphPFN import 失败

确保在正确的目录运行，并且 `lib/graphpfn` 存在:
```bash
cd /home/figurich/CodeProjects/graph-dexposure
ls lib/graphpfn/model.py
```

### Q: Terra/FTX 事件不在数据范围内

检查数据日期范围:
```python
import json
with open("data/historical-network_week_2025-07-01.json") as f:
    data = json.load(f)
dates = sorted(data["data"].keys())
print(f"Date range: {dates[0]} to {dates[-1]}")
```

---

## 联系方式

如有问题，可以查看:
- 论文草稿: `DeXposure_FM.pdf`
- 实验计划: `experiment-plan.md`

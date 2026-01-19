# DeXposure-FM 实验指南

> 最后更新: 2026-01-19

## 项目目标

为 NBER "AI and Economic Measurement" 会议准备论文实验，验证 GraphPFN 在 DeFi 协议间信用暴露预测任务上的表现。

### 导师反馈摘要 (2026-01-19)

| 类别 | 反馈 | 状态 |
|------|------|------|
| 架构 | GraphPFN 单编码器方案 | ✅ 认可 |
| 训练 | 需实现 7个损失分量 | 🔴 待实现 |
| 训练 | 需实现 TVL 经济加权 | 🔴 待实现 |
| 实验 | 保留 h ∈ {1,3,7,14} 窗口 | 🔴 h=14 待补 |

### 论文核心评估任务

| 任务 | 描述 | 状态 |
|------|------|------|
| Task I | Multi-step Forecasting (h=1,3,7,14) | ⏳ 缺 h=14 |
| Task II | Shock Analysis (Terra/FTX) | ✅ 已实现 |
| Task III | Imputation (缺失值填补) | ✅ 已实现 |
| Ablation | 损失分量消融 (7个) | 🔴 待实现 |
| Ablation | TVL 加权消融 | 🔴 待实现 |
| Ablation | 编码器消融 (GraphPFN vs GNN) | ⏳ 待补充 |

---

## 🔧 代码优化

### 已完成
| 优先级 | 任务 | 说明 | 状态 |
|--------|------|------|------|
| 高 | 分层学习率 | GraphPFN encoder 用低 lr (1e-4)，task heads 用高 lr (1e-3) | ✅ 已实现 |
| 低 | 中间结果保存 | 每个horizon完成后保存metrics_intermediate.json | ✅ 已实现 |
| 低 | ROLAND GRU修复 | 修复不同节点数图之间的hidden state传递bug | ✅ 已实现 |

### 待实现 (导师要求)
| 优先级 | 任务 | 说明 | 状态 |
|--------|------|------|------|
| 🔴 高 | 7个损失分量 | L_edge, L_link, L_node, L_stats, L_impute, L_scen, L_smooth | ⬜ 待实现 |
| 🔴 高 | TVL 经济加权 | $w_{ij} \propto TVL_i \cdot E_{ij}$ | ⬜ 待实现 |
| 🔴 高 | h=14 预测窗口 | 添加两周预测实验 | ⬜ 待运行 |
| 🟡 中 | 编码器消融 | GraphPFN vs GCN/GAT/SAGE | ⬜ 待实现 |
| 🟡 中 | Naive Baseline | 用上周边直接预测 | ⬜ 待实现 |
| 低 | 渐进解冻 | 先 frozen 训练几个 epoch，再 finetune | ⬜ 待实现 |
| 低 | 使用 AdamW | 替换 Adam，添加 weight decay | ⬜ 待实现 |

**参考实现 (分层学习率):**
```python
if finetune:
    optimizer = torch.optim.AdamW([
        {"params": model.encoder.parameters(), "lr": config.lr * 0.1},  # 骨干: 低 lr
        {"params": model.link_scorer.parameters(), "lr": config.lr},   # 新头: 高 lr
        {"params": model.node_head.parameters(), "lr": config.lr},
    ], weight_decay=config.weight_decay)
```

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
GraphPFN (Frozen)         0.9308      0.9343      0.9337      3.27         4.33e-05
GraphPFN (Finetuned)      待运行       待运行       待运行       待运行        待运行
ROLAND                    待运行       待运行       待运行       N/A          待运行

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

output/graph-dexposure-results/2025-01-16_graphpfn_frozen/
├── experiment_results.json       # GraphPFN Frozen 完整结果
├── data_quality.json             # 283周数据质量
├── network_statistics.csv        # 网络演变统计
├── README.md                     # 结果摘要
└── graphpfn_frozen/
    ├── predictions_edges_test.csv  # 边预测 (1.3GB)
    └── predictions_nodes_test.csv  # 节点预测 (51MB)
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

## 🎯 完整实验结果 (2025-01-17)

### 实验配置 (所有模型)

| 参数 | 值 |
|------|-----|
| **方法** | Expanding Window Walk-Forward |
| **Folds数量** | 16 |
| **训练周数** | 238周 (2020-03-23 ~ 2024-10-07) |
| **验证周数** | 12周 (2024-10-14 ~ 2024-12-30) |
| **测试周数** | 33周 (2025-01-06 ~ 2025-08-18) |
| **硬件** | NVIDIA H100 (80GB) |
| **训练轮数** | 5 epochs |
| **分层学习率** | Encoder: 1e-4, Head: 1e-3 (Finetuned模式) |

---

### 📊 模型对比 - Multi-step Forecasting Results

| Model | h=1 AUPRC | h=1 AUROC | h=3 AUPRC | h=3 AUROC | h=7 AUPRC | h=7 AUROC | Weight MAE |
|-------|-----------|-----------|-----------|-----------|-----------|-----------|------------|
| **GraphPFN (Finetuned)** | **0.9322** | **0.9855** | **0.9344** | **0.9860** | **0.9361** | **0.9861** | 2.62-2.66 |
| **GraphPFN (Frozen)** | 0.9308 | 0.9863 | 0.9343 | 0.9867 | 0.9337 | 0.9861 | 3.25-3.30 |
| **ROLAND** | 0.8697 | 0.9616 | 0.8655 | 0.9593 | 0.8614 | 0.9550 | 3.94-4.00 |

**关键发现:**
- ✅ **GraphPFN Finetuned 在所有 horizons 上最优**: AUPRC 0.932-0.936
- ✅ **Finetuned 比 Frozen 提升 0.1-0.3% AUPRC**: 分层学习率微调有效
- ✅ **GraphPFN 比 ROLAND 提升 7-9% AUPRC**: Foundation model 优势明显
- ✅ **h=7 性能最佳**: 长期预测鲁棒，AUPRC 0.9361 (最高)
- ✅ **Weight MAE**: Finetuned (2.62) < Frozen (3.25) < ROLAND (3.99)

**详细结果:**

#### GraphPFN (Frozen) - 线性探针

| Horizon | AUPRC | AUROC | Weight MAE | Weighted MAE | Recall@100 |
|---------|-------|-------|------------|--------------|------------|
| h=1 | 0.9308 | 0.9863 | 3.3037 | 13.78 | 4.33e-05 |
| h=3 | 0.9343 | 0.9867 | 3.2680 | 13.44 | 4.63e-05 |
| h=7 | 0.9337 | 0.9861 | 3.2549 | 13.75 | 5.37e-05 |

#### GraphPFN (Finetuned) - 分层学习率微调

| Horizon | AUPRC | AUROC | Weight MAE | Weighted MAE | Recall@100 |
|---------|-------|-------|------------|--------------|------------|
| h=1 | 0.9322 | 0.9855 | 2.6217 | 6.10 | 4.33e-05 |
| h=3 | 0.9344 | 0.9860 | 2.6645 | 5.97 | 4.63e-05 |
| h=7 | 0.9361 | 0.9861 | 2.6441 | 6.24 | 5.37e-05 |
| h=14 | ⏳ 待运行 | ⏳ | ⏳ | ⏳ | ⏳ |

**分层学习率配置:**
- Encoder (预训练): LR = 1e-4 (0.1x)
- Head (新任务): LR = 1e-3 (1.0x)

#### ROLAND Baseline

| Horizon | AUPRC | AUROC | Weight MAE | Weighted MAE | Recall@100 |
|---------|-------|-------|------------|--------------|------------|
| h=1 | 0.8697 | 0.9616 | 3.9878 | 18.98 | 4.33e-05 |
| h=3 | 0.8655 | 0.9593 | 3.9682 | 19.00 | 4.58e-05 |
| h=7 | 0.8614 | 0.9550 | 3.9377 | 19.08 | 5.21e-05 |

### 📈 网络统计摘要

| 指标 | 均值 | 标准差 | 最小值 | 最大值 |
|------|------|--------|--------|--------|
| 节点数 | 5,676 | 3,952 | 9 | 11,236 |
| 边数 | 30,424 | 25,589 | 8 | 74,138 |
| 度Gini系数 | 0.703 | 0.071 | 0.389 | 0.748 |
| TVL Gini | 0.986 | 0.011 | 0.889 | 0.992 |
| 跨部门暴露比 | 0.853 | 0.151 | 0.0 | 0.979 |
| 总TVL (B USD) | 276.1 | 220.3 | 0.09 | 858.7 |

### 📁 输出文件

结果保存在 `output/graph-dexposure-results/2025-01-16_graphpfn_frozen/`:
```
├── experiment_results.json     # 完整实验结果 (5869行)
├── data_quality.json           # 数据质量统计 (2563行)
├── network_statistics.csv      # 网络统计时序数据 (283周)
├── experiment.log              # 实验日志
├── README.md                   # 结果摘要
└── graphpfn_frozen/
    ├── predictions_edges_test.csv  # 边级预测 (1.3GB)
    └── predictions_nodes_test.csv  # 节点级预测 (51MB)
```

### 🚀 已完成的实验

- ✅ **GraphPFN Frozen**: 线性探针 baseline
- ✅ **GraphPFN Finetuned**: 分层学习率微调
- ✅ **ROLAND**: Temporal GNN baseline
- ✅ **Network Statistics**: 网络统计量计算

### 📋 待运行实验

**🔴 高优先级 (导师要求):**
1. **h=14 预测窗口**: 补充两周预测实验
2. **7个损失分量**: 实现并消融验证
3. **TVL 经济加权**: 实现并消融验证

**🟡 中优先级:**
4. **编码器消融**: GraphPFN vs GCN/GAT/SAGE
5. **Naive Baseline**: 验证模型优于简单启发式
6. **Shock Analysis**: Terra/FTX 事件分析
7. **Imputation**: 缺失值填补实验

**低优先级:**
8. **统计显著性测试**: 多种子运行 (seeds: 42, 123, 456, 789, 2024)

---

## 之前的实验结果

### 初步测试 (单次运行)

使用 `run_dexposure_experiment.py` 的单次运行结果:

```
AUPRC: 0.814
AUROC: 0.958
Weight MAE: 1.627
```

> **注意**: 最新的完整实验 (2025-01-16) 使用更严格的时序划分和更大的测试集，获得了更好的 AUPRC (0.93+) 和 AUROC (0.98+) 结果。

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

# DeXposure-FM 实验指南

> 最后更新: 2025-01-07

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
cd /home/figurich/inter-protocol-exposure/graphpfn
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
# 网络数据 (约200MB)
ls -lh data/historical-network_week_2025-07-01.json

# 元数据
ls -lh data/meta_df.csv

# GraphPFN 预训练checkpoint
ls -lh checkpoints/graphpfn-v1.ckpt
```

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

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--mode` | all | 实验模式 |
| `--epochs` | 5 | 训练轮数 |
| `--seed` | 42 | 随机种子 |
| `--output-dir` | output/full_experiment | 输出目录 |
| `--shock-model` | graphpfn | Shock分析使用的模型 |

---

## 预期输出

### 1. 终端输出示例

```
============================================================
COMPARISON TABLE - Multi-step Forecasting Results
============================================================
Model                          h=1 AUPRC       h=3 AUPRC       h=7 AUPRC
--------------------------------------------------------------------------------
GraphPFN (Frozen)              0.7832          0.7456          0.7123
GraphPFN (Finetuned)           0.8142          0.7821          0.7534
ROLAND                         0.7234          0.6987          0.6654

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
├── experiment_results.json    # 完整实验结果
└── network_statistics.csv     # 网络统计时序数据
```

### 3. 关键指标解读

**Link Prediction:**
- AUPRC > 0.8 表示良好的预测能力
- AUROC > 0.9 表示优秀的区分能力

**Network Statistics:**
- Gini: 0=完全平等, 1=完全不平等
- HHI: 0=分散, 1=集中
- Density: 边数/最大可能边数

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
graphpfn/
├── run_full_experiment.py      # 主实验脚本 (~1850行)
├── run_optuna_search.py        # Optuna 超参数优化
├── run_final_evaluation.py     # 最终评估 (2025 hold-out test)
├── run_dexposure_experiment.py # 单次实验脚本 (已验证可用)
├── EXPERIMENT_PLAN.md          # 详细实验计划 (给老师看)
├── src/
│   └── network_statistics.py   # 网络统计量模块
├── data/
│   ├── historical-network_week_2025-07-01.json
│   └── meta_df.csv
├── checkpoints/
│   └── graphpfn-v1.ckpt
└── output/
    ├── full_experiment/        # 完整实验结果
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

| 论文Section | 实验内容 | 脚本/函数 |
|-------------|----------|-----------|
| 5.2 Multi-step | h=1,3,7预测 | `run_graphpfn_experiment()` |
| 5.2 Baselines | Frozen vs Finetuned vs ROLAND | `--mode frozen/finetuned/roland` |
| 5.3 Shock Analysis | Terra/FTX事件 | `run_shock_analysis()` |
| 5.4 Network Stats | Gini/HHI/Density | `run_network_statistics()` |
| 5.5 Imputation | 缺失值重建 | `run_imputation_experiment()` |
| Appendix | Rolling evaluation | `rolling_evaluation()` |

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
cd /home/figurich/inter-protocol-exposure/graphpfn
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

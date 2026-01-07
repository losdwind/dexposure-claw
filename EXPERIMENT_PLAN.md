# DeXposure-FM 实验计划

> 版本: v2.0 | 日期: 2025-01-07

## 0. 研究目标与核心对照

### 0.1 研究目标

验证 Graph-Tabular Foundation Model (GraphPFN) 在 DeFi 协议间信用暴露 (Inter-Protocol Exposure) 预测任务上的有效性。

**主线叙事 (NBER):**
> 我们把 DeFi 系统视为"时序信用敞口网络"。**机制层**是协议间敞口边（edge exposures）的形成与演化；**状态层**是协议规模（TVL proxy）变化。
> 因此以 **edge-level exposure forecasting（exist + weight）为主任务**，以 **node-level TVL log-change 为辅助任务**进行多任务训练。

### 0.2 核心对照实验

| 对照组 | Encoder | Scorer | 说明 |
|--------|---------|--------|------|
| **GraphPFN (Frozen Probing)** | 冻结 | 训练 | 衡量预训练表示的可迁移性 |
| **GraphPFN (Finetuned)** | 微调 | 训练 | 衡量 DeXposure 微调增益 |
| **ROLAND (done in Dexposure)** | 从头训练 | 训练 | 传统时序GNN基线 |

### 0.3 最终交付物

- `metrics.json`: 各模型在 Test 的 AUPRC/AUROC、MAE/RMSE
- `predictions_edges_test.csv`: 边级预测（存在性 + 边权）
- `predictions_nodes_test.csv`: 节点 TVL log-change 预测
- `data_quality.json`: 每周数据质量与缺失统计
- 论文 Table 1–5 + Figure 1–7

---

## 1. Neural Architecture

### 1.1 GraphPFN (主模型)

GraphPFN 是一个预训练的 Graph-Tabular Foundation Model，基于 Transformer 架构，支持 in-context learning。

```
┌─────────────────────────────────────────────────────────────┐
│                    GraphPFN Architecture                     │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  Input: Node Features (X) + Graph Structure (A)             │
│           ↓                                                 │
│  ┌─────────────────────────────────────────────┐           │
│  │     GraphPFN Transformer Encoder            │           │
│  │  ┌─────────────────────────────────────┐   │           │
│  │  │ Multi-head Self-Attention (MHA)     │   │           │
│  │  │ + Graph Structure Injection         │   │           │
│  │  │ + Positional Encoding               │   │           │
│  │  └─────────────────────────────────────┘   │           │
│  │              × L layers                     │           │
│  └─────────────────────────────────────────────┘           │
│           ↓                                                 │
│  Node Embeddings H ∈ ℝ^{N × d}                             │
│           ↓                                                 │
│  ┌─────────────────────────────────────────────┐           │
│  │         Task-specific Heads                 │           │
│  │  • Link Scorer: [h_u; h_v; h_u⊙h_v; |h_u-h_v|] → MLP   │
│  │  • Node Head: h_i → MLP → Δ_size            │           │
│  └─────────────────────────────────────────────┘           │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

**架构参数:**

| 参数 | 值 | 说明 |
|------|-----|------|
| `embed_dim` | 256 | Transformer embedding dimension |
| `num_layers` | 12 | Number of transformer layers |
| `num_heads` | 8 | Multi-head attention heads |
| `ffn_dim` | 1024 | Feed-forward network dimension |
| `dropout` | 0.1 | Dropout rate |
| `max_nodes` | 1000 | Maximum nodes per graph |

**Link Scorer Head:**
```python
class LinkScorer(nn.Module):
    # Input: 4 * embed_dim (concatenation of [h_u, h_v, h_u⊙h_v, |h_u-h_v|])
    # Hidden: 256
    # Output: exist_logit (1), weight_pred (1)
```

**Node Prediction Head:**
```python
class NodeHead(nn.Module):
    # Input: embed_dim
    # Hidden: 256
    # Output: Δlog(TVL) (1)
```

### 1.2 ROLAND Baseline

Temporal GNN baseline with GCN encoder and GRU temporal update.

```
┌─────────────────────────────────────────────────────────────┐
│                    ROLAND Architecture                       │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  Input: X_t, A_t, H_{t-1}                                   │
│           ↓                                                 │
│  ┌─────────────────────────────────────────────┐           │
│  │  MLP Preprocessing                          │           │
│  │  Linear(input_dim, 128) → LeakyReLU → DO    │           │
│  │  Linear(128, 64) → LeakyReLU → DO           │           │
│  └─────────────────────────────────────────────┘           │
│           ↓                                                 │
│  ┌─────────────────────────────────────────────┐           │
│  │  GCN Layer 1 + GRU Update 1                 │           │
│  │  h = GCN(h, A) → GRU(h, h1_prev)            │           │
│  └─────────────────────────────────────────────┘           │
│           ↓                                                 │
│  ┌─────────────────────────────────────────────┐           │
│  │  GCN Layer 2 + GRU Update 2                 │           │
│  │  h = GCN(h, A) → GRU(h, h2_prev)            │           │
│  └─────────────────────────────────────────────┘           │
│           ↓                                                 │
│  Link Prediction: h_u ⊙ h_v → Linear → sigmoid             │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

**架构参数:**

| 参数 | 值 | 说明 |
|------|-----|------|
| `hidden_dim` | 64 | GCN hidden dimension |
| `out_dim` | 32 | Output embedding dimension |
| `num_gcn_layers` | 2 | Number of GCN layers |
| `dropout` | 0.1 | Dropout rate |

---

## 2. Dataset

### 2.1 数据来源

- **来源**: DeXposure (基于 DeFiLlama API)
- **时间范围**: 2020-03 ~ 2025-08 (283周)
- **粒度**: Weekly snapshots
- **元数据**: `data/meta_df.csv` (id, name, category)

**数据文件:**

| 文件 | 大小 | 周数 | 日期范围 |
|------|------|------|----------|
| `historical-network_week_2020-03-30.json` | 1.1GB | 283 | 2020-03-23 ~ 2025-08-18 |
| `historical-network_week_2025-07-01.json` | 76MB | 8 | 2025-06-30 ~ 2025-08-18 |

> **注意**: 主实验使用 1.1GB 完整历史文件。76MB 文件仅包含最近快照。

### 2.2 JSON 快照结构

每个周 `YYYY-MM-DD` 包含:
```json
{
  "YYYY-MM-DD": {
    "nodes": [{"id": "protocol_id", "size": 1234567, "composition": {...}}],
    "links": [{"source": "A", "target": "B", "size": 12345, "composition": {...}}]
  }
}
```

### 2.3 数据清洗规则

| 规则 | 说明 | 记录 |
|------|------|------|
| 丢弃 `target == null` 的边 | 外部资产,非协议间暴露 | `pct_target_null_dropped` |
| 丢弃端点不在节点集合的边 | 数据一致性 | `pct_endpoint_missing_dropped` |
| `category` join 不到 | 置为 `"Unknown"` | `pct_category_unknown` |

### 2.4 数据质量统计 (写入 data_quality.json)

每周记录:
```python
{
    "date": "YYYY-MM-DD",
    "N_nodes": int,
    "N_edges": int,
    "pct_target_null_dropped": float,
    "pct_endpoint_missing_dropped": float,
    "overlap_ratio_next_week": float,  # |V_t ∩ V_{t+1}| / |V_t|
    "pct_category_unknown": float
}
```

### 2.5 数据统计

| 指标 | 值 |
|------|-----|
| Total snapshots | ~280 weeks |
| Nodes per snapshot | 50-500 protocols |
| Edges per snapshot | 100-2000 exposure links |
| Node features | 4 + |categories| (约15维) |
| Edge weight range | $1K - $10B (log-scaled) |

### 2.6 Node Features (特征工程)

| Feature | Formula | Description |
|---------|---------|-------------|
| `log_size` | `log1p(size_t(u))` | Protocol TVL (log-scaled) |
| `num_tokens` | `len(composition)` | Number of token types |
| `max_share` | `max(comp.values) / (size + eps)` | Largest token share, eps=1e-9 |
| `entropy` | `-Σ p_k log(p_k + eps)` | Token composition entropy |
| `category_onehot` | One-hot encoding | Protocol category (~15 classes) |

**Entropy 计算细节:**
```python
eps = 1e-9
p_k = value_k / (sum(values) + eps)
entropy = -sum(p_k * np.log(p_k + eps) for p_k in probs if p_k > 0)
```

### 2.7 Train/Test Split — Expanding Window Walk-Forward (金融惯例)

**原则**:
1. 测试集数据必须在时间上完全晚于训练集，确保无数据泄露
2. 使用 Expanding Window 方法进行多次 out-of-sample 评估
3. 保留 2025 数据作为最终 Hold-out Test Set

```
Timeline (Expanding Window Walk-Forward):
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

**默认参数配置:**

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `min_train_weeks` | 104 | 最少训练周数 (2年) |
| `val_weeks` | 12 | 验证集大小 (约3个月) |
| `test_weeks` | 8 | 每fold测试大小 (约2个月) |
| `step_weeks` | 8 | 滚动步长 |
| `holdout_start` | 2025-01-01 | Hold-out 测试集起始日期 |

**数据划分示例 (283周数据):**

| 划分 | 时间范围 | 周数 |
|------|----------|------|
| **Rolling Folds** | 2020-03 ~ 2024-12 | 约15个folds |
| **Hold-out Train** | 2020-03-23 ~ 2024-09-30 | ~238周 |
| **Hold-out Val** | 2024-10-07 ~ 2024-12-30 | 12周 |
| **Hold-out Test** | 2025-01-06 ~ 2025-08-18 | ~33周 |

**运行命令:**

```bash
# 快速实验 (仅 Hold-out 评估)
python run_full_experiment.py --mode all --epochs 10

# 完整 Rolling Window 评估 (更稳健，但更慢)
python run_full_experiment.py --mode all --epochs 10 --rolling
```

**为什么用 Expanding Window?**

1. **金融惯例**: 避免 look-ahead bias，符合实际交易场景
2. **多次评估**: 不依赖单一测试集，结果更稳健
3. **训练数据增长**: 随时间积累更多数据，模拟真实部署
4. **Hold-out 验证**: 2025 数据从未用于任何训练/调参

---

## 3. Tasks (数学定义)

给定周 t 的图 $G_t = (V_t, E_t)$ 与特征，预测周 t+h。

### 3.1 Task I: Multi-step Forecasting

**目标**: 预测未来 h 周的网络结构变化

| Horizon | Description |
|---------|-------------|
| h = 1 | Next week prediction |
| h = 3 | 3 weeks ahead |
| h = 7 | 7 weeks ahead |

**Sub-task A: Edge Existence (分类)**
$$y^{exist}_{t+h}(u,v) = \mathbb{1}[(u,v) \in E_{t+h}]$$

**Sub-task B: Edge Weight (回归,仅正例边)**
$$y^{w}_{t+h}(u,v) = \log(1 + \text{size}_{t+h}(u,v))$$

**Sub-task C: Node TVL log-change (辅助回归)**
$$y^{node}_{t+h}(u) = \log(1 + \text{size}_{t+h}(u)) - \log(1 + \text{size}_t(u))$$

> 仅对 $V_t \cap V_{t+h}$ 计算标签

### 3.2 Task II: Shock Analysis (Policy-relevant Scenarios)

**目标**: 评估模型在极端市场事件期间的表现

| Event | Date | Analysis Window |
|-------|------|-----------------|
| Terra/Luna Collapse | 2022-05-09 | 2022-04-25 ~ 2022-05-23 |
| FTX Collapse | 2022-11-07 | 2022-10-31 ~ 2022-11-21 |

**评估内容:**
- 事件前后网络结构变化 (TVL, edges, Gini, HHI)
- 模型预测性能退化程度
- 是否能提前预警风险

### 3.3 Task III: Imputation

**目标**: 测试模型的缺失值重建能力

| Mask Ratio | Description |
|------------|-------------|
| 10% | Light masking |
| 20% | Moderate masking |
| 30% | Heavy masking |

**Masking Types:**
- Edge masking: 随机移除部分边
- Node masking: 随机mask节点属性
- Combined masking: 同时mask边和节点

---

## 4. Training Sample Construction

### 4.1 周对样本

样本单位：每个周对 `(t -> t+h)`
- 训练输入用 $G_t$
- 标签来自 $t+h$

### 4.2 正例边 (Positive Edges)

```python
pos_edges = E_{t+h}  # 已清洗
```

### 4.3 负采样 (Negative Sampling)

**默认策略: uniform, neg:pos = 5:1**

```python
neg_ratio = 5  # 5 negative samples per positive edge

# Sampling procedure:
# 1. 从 V_t × V_t (有向) 随机采样 (u, v)
# 2. 约束: (u, v) 不在 pos_edges
# 3. 采样数: |neg| = neg_ratio * |pos|
# 4. 固定随机种子,保证可复现
```

**稳健性测试 (Appendix):**
- degree-biased negatives
- 更高 test neg 比例 (10:1, 50:1)

### 4.4 Pairs 列表 (用于存在性训练)

```python
pairs = pos_edges ∪ neg_edges
label_exist ∈ {0, 1}
label_weight = log1p(size_{t+h}(u,v))  # 仅正例; 负例 weight mask=False
```

### 4.5 Node 标签与 Mask (辅助任务)

```python
# 对共同节点 V_t ∩ V_{t+h}:
y_node(u) = log1p(size_{t+h}) - log1p(size_t)
node_mask = True  # 标识哪些节点有标签
```

---

## 5. Loss Functions

### 5.1 总损失函数

```python
L_total = λ_exist * L_exist + λ_weight * L_weight + λ_node * L_node
```

**默认权重:**

| 损失项 | 权重 (λ) | 说明 |
|--------|----------|------|
| `L_exist` | 1.0 | Edge existence loss |
| `L_weight` | 1.0 | Edge weight loss |
| `L_node` | 0.5 | Node size change loss |

### 5.2 各损失函数定义

**Link Existence Loss (Binary Cross-Entropy):**
```python
L_exist = BCE(σ(logits), y_exist)
       = -[y·log(σ(logits)) + (1-y)·log(1-σ(logits))]
```

**Edge Weight Loss (Smooth L1 / Huber Loss):**
```python
L_weight = SmoothL1(w_pred, log1p(w_true))

# Only computed for positive edges (existing edges)
# SmoothL1 is robust to outliers
```

**Node Size Change Loss (Smooth L1):**
```python
L_node = SmoothL1(Δsize_pred, Δsize_true)

# Where Δsize = log1p(size_{t+h}) - log1p(size_t)
```

### 5.4 Ablation: 移除 Node Aux Task

```python
# 设置 λ_node = 0, 观察主任务是否下降
L_total = 1.0 * L_exist + 1.0 * L_weight + 0.0 * L_node
```

---

## 6. Evaluation Metrics

### 6.1 Link Existence Metrics

| Metric | Formula | Description |
|--------|---------|-------------|
| **AUPRC** | Area under PR curve | Primary metric (handles class imbalance) |
| **AUROC** | Area under ROC curve | Secondary metric |
| **F1@0.5** | F1 at threshold 0.5 | Practical threshold metric |

### 6.2 Edge Weight Metrics

| Metric | Formula | Description |
|--------|---------|-------------|
| **MAE** | mean(\|y - ŷ\|) | Mean Absolute Error |
| **RMSE** | sqrt(mean((y - ŷ)²)) | Root Mean Squared Error |
| **R²** | 1 - SS_res/SS_tot | Coefficient of determination |

### 6.3 Node Size Metrics

| Metric | Formula | Description |
|--------|---------|-------------|
| **MAE** | mean(\|Δy - Δŷ\|) | Mean Absolute Error |
| **RMSE** | sqrt(mean((Δy - Δŷ)²)) | Root Mean Squared Error |

### 6.4 Tail / 系统性重要性指标

| Metric | Formula | Description |
|--------|---------|-------------|
| **Recall@K** | Recall at top K | K=100/500/1000 或 top 0.1%/1% |
| **Weighted MAE** | 按真实边权加权的MAE | 对大额暴露更敏感 |

### 6.5 Network-level Metrics

| Metric | Description |
|--------|-------------|
| **Gini Coefficient** | Degree distribution inequality |
| **HHI** | Herfindahl-Hirschman Index (concentration) |
| **Density** | Edge density |
| **Entropy** | Degree distribution entropy |
| **Top-10% Concentration** | Share of top 10% nodes |

---

## 7. Optimizer & Hyperparameters

### 7.1 默认超参数配置

```python
@dataclass
class ExperimentConfig:
    # Optimizer
    optimizer: str = "Adam"
    lr: float = 1e-3
    weight_decay: float = 1e-4

    # Training
    epochs: int = 10
    batch_size: int = 20000  # edge batch size

    # Data
    neg_ratio: int = 5
    train_ratio: float = 0.60
    val_ratio: float = 0.20
    test_ratio: float = 0.20

    # Model
    hidden_dim: int = 256

    # Loss weights
    exist_loss_weight: float = 1.0
    weight_loss_weight: float = 1.0
    node_loss_weight: float = 0.5

    # Rolling evaluation
    rolling_window_size: int = 52
    rolling_stride: int = 4

    # Random seeds for significance testing
    random_seeds: List[int] = [42, 123, 456, 789, 2024]
```

### 7.2 Optuna 超参数优化

```python
import optuna

def objective(trial):
    # Hyperparameter search space
    config = ExperimentConfig(
        lr=trial.suggest_loguniform("lr", 1e-5, 1e-2),
        weight_decay=trial.suggest_loguniform("weight_decay", 1e-6, 1e-3),
        hidden_dim=trial.suggest_categorical("hidden_dim", [128, 256, 512]),
        neg_ratio=trial.suggest_int("neg_ratio", 3, 10),
        exist_loss_weight=trial.suggest_float("exist_loss_weight", 0.5, 2.0),
        weight_loss_weight=trial.suggest_float("weight_loss_weight", 0.5, 2.0),
        node_loss_weight=trial.suggest_float("node_loss_weight", 0.1, 1.0),
        epochs=trial.suggest_int("epochs", 5, 20),
    )

    # Train and evaluate
    result = run_graphpfn_experiment(config, finetune=True)

    # Optimize for AUPRC on validation set
    return result["results"]["h1"]["exist"]["auprc"]

# Run optimization
study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=50, timeout=3600*6)  # 6 hours

print("Best hyperparameters:", study.best_params)
```

**搜索空间:**

| Hyperparameter | Search Space | Scale |
|----------------|--------------|-------|
| `lr` | [1e-5, 1e-2] | Log-uniform |
| `weight_decay` | [1e-6, 1e-3] | Log-uniform |
| `hidden_dim` | {128, 256, 512} | Categorical |
| `neg_ratio` | [3, 10] | Integer |
| `exist_loss_weight` | [0.5, 2.0] | Uniform |
| `weight_loss_weight` | [0.5, 2.0] | Uniform |
| `node_loss_weight` | [0.1, 1.0] | Uniform |
| `epochs` | [5, 20] | Integer |

### 7.3 Learning Rate Schedule

```python
# Option 1: Constant LR (default)
optimizer = Adam(params, lr=1e-3)

# Option 2: Cosine Annealing (for longer training)
scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)

# Option 3: ReduceLROnPlateau (adaptive)
scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=3)
```

---

## 8. SOTA Comparison

### 8.1 Baseline Methods

| Method | Type | Description |
|--------|------|-------------|
| **GraphPFN (Frozen)** | Foundation Model | Pretrained encoder + linear probe |
| **GraphPFN (Finetuned)** | Foundation Model | End-to-end fine-tuning |
| **ROLAND** | Temporal GNN | GCN + GRU (You et al., 2022) |
| **EvolveGCN** | Temporal GNN | Evolving GCN weights |
| **DySAT** | Temporal GNN | Dynamic self-attention |

### 8.2 预期结果对比表

| Model | h=1 AUPRC | h=3 AUPRC | h=7 AUPRC | Weight MAE |
|-------|-----------|-----------|-----------|------------|
| GraphPFN (Finetuned) | **0.82+** | **0.78+** | **0.75+** | **1.5** |
| GraphPFN (Frozen) | 0.78 | 0.74 | 0.71 | 1.8 |
| ROLAND | 0.72 | 0.68 | 0.65 | 2.1 |
| EvolveGCN | 0.70 | 0.66 | 0.63 | 2.3 |
| DySAT | 0.71 | 0.67 | 0.64 | 2.2 |

### 8.3 统计显著性测试

```python
# Run each model with 5 different seeds
seeds = [42, 123, 456, 789, 2024]

# Compute mean ± std
results = {model: [] for model in models}
for seed in seeds:
    for model in models:
        result = run_experiment(model, seed=seed)
        results[model].append(result["auprc"])

# Report: mean ± std
for model in models:
    print(f"{model}: {np.mean(results[model]):.4f} ± {np.std(results[model]):.4f}")

# Paired t-test for significance
from scipy.stats import ttest_rel
t_stat, p_value = ttest_rel(results["GraphPFN"], results["ROLAND"])
print(f"p-value: {p_value:.4f}")
```

---

## 9. 实验执行计划

### 快速开始

```bash
# 运行完整实验 (严格时间划分: 2020-2024 train, 2025 test)
python run_full_experiment.py --mode all --epochs 10 --save-predictions

# 输出:
#   - output/full_experiment/experiment_results.json
#   - output/full_experiment/data_quality.json
#   - output/full_experiment/predictions_edges_*.csv
#   - output/full_experiment/predictions_nodes_*.csv
```

### Phase 1: 基础验证

```bash
# 1. 单次运行验证代码正确性 (使用严格时间划分)
python run_full_experiment.py --mode finetuned --epochs 5

# 2. 检查输出格式和指标
cat output/full_experiment/experiment_results.json
cat output/full_experiment/data_quality.json
```

### Phase 2: 完整对比实验

```bash
# 1. 运行所有模型对比 (Frozen/Finetuned/ROLAND)
python run_full_experiment.py --mode all --epochs 10 --save-predictions

# 2. 多种子运行 (统计显著性测试)
for seed in 42 123 456 789 2024; do
    python run_full_experiment.py --mode all --seed $seed --output-dir output/seed_$seed
done
```

### Phase 3: 超参数优化 (可选)

```bash
# 运行 Optuna 优化
python run_optuna_search.py --n-trials 50 --timeout 21600
```

### 命令行参数说明

**实验设置:**

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--mode` | all | frozen/finetuned/roland/stats/shock/impute/all |
| `--epochs` | 5 | 训练轮数 |
| `--seed` | 42 | 随机种子 |
| `--output-dir` | output/full_experiment | 输出目录 |
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

---

## 10. 预期产出

### 10.1 实验结果表格

**Table 1: Multi-step Forecasting Results**

| Model | h=1 AUPRC | h=1 AUROC | h=3 AUPRC | h=7 AUPRC | Weight MAE |
|-------|-----------|-----------|-----------|-----------|------------|
| GraphPFN (FT) | 0.82±0.02 | 0.96±0.01 | 0.78±0.02 | 0.75±0.03 | 1.5±0.1 |
| GraphPFN (Frozen) | 0.78±0.02 | 0.94±0.01 | 0.74±0.02 | 0.71±0.03 | 1.8±0.1 |
| ROLAND | 0.72±0.03 | 0.91±0.02 | 0.68±0.03 | 0.65±0.03 | 2.1±0.2 |

**Table 2: Shock Analysis Results**

| Event | Model | Pre-AUPRC | Event-AUPRC | Degradation |
|-------|-------|-----------|-------------|-------------|
| Terra | GraphPFN | 0.82 | 0.74 | -0.08 |
| Terra | ROLAND | 0.72 | 0.61 | -0.11 |
| FTX | GraphPFN | 0.81 | 0.76 | -0.05 |
| FTX | ROLAND | 0.71 | 0.63 | -0.08 |

---

## 11. 输出文件规范

### 11.1 predictions_edges_test.csv

```csv
time_t,time_t1,u_id,v_id,y_exist_true,y_exist_pred,y_w_true,y_w_pred,is_positive
2024-12-30,2025-01-06,aave,compound,1,0.87,15.23,14.89,True
2024-12-30,2025-01-06,uniswap,curve,0,0.12,0,0,False
...
```

### 11.2 predictions_nodes_test.csv

```csv
time_t,time_t1,node_id,y_node_true,y_node_pred,size_t,category
2024-12-30,2025-01-06,aave,0.023,-0.015,1234567890,Lending
...
```

### 11.3 metrics.json

```json
{
  "model": "GraphPFN (Finetuned)",
  "h1": {
    "exist": {"auprc": 0.814, "auroc": 0.958},
    "weight": {"mae": 3.23, "rmse": 4.05},
    "node": {"mae": 0.125}
  },
  "h3": {...},
  "h7": {...}
}
```

### 11.4 data_quality.json

```json
{
  "summary": {
    "mean_nodes_per_week": 245,
    "mean_edges_per_week": 1234,
    "pct_target_null_dropped": 0.15,
    "mean_overlap_ratio": 0.92
  },
  "weekly": [...]
}
```

---

## 12. 论文图表模板

### Table 1: Main Results - Frozen vs Finetuned

| Model | Encoder | Scorer | Exist AUPRC ↑ | Exist AUROC ↑ | Weight MAE ↓ | Recall@K ↑ | Node MAE ↓ |
|-------|---------|--------|--------------|---------------|--------------|------------|------------|
| GraphPFN (Frozen) | frozen | trained | [TBD] | [TBD] | [TBD] | [TBD] | [TBD] |
| **GraphPFN (FT)** | finetuned | trained | **0.814** | **0.958** | **3.23** | [TBD] | **0.125** |
| ROLAND | trained | trained | [TBD] | [TBD] | [TBD] | [TBD] | [TBD] |

### Table 2: Multi-step Forecasting (h=1,3,7)

| Model | h=1 AUPRC | h=3 AUPRC | h=7 AUPRC | Weight MAE |
|-------|-----------|-----------|-----------|------------|
| GraphPFN (FT) | 0.82±0.02 | 0.78±0.02 | 0.75±0.03 | 1.5±0.1 |
| GraphPFN (Frozen) | 0.78±0.02 | 0.74±0.02 | 0.71±0.03 | 1.8±0.1 |
| ROLAND | 0.72±0.03 | 0.68±0.03 | 0.65±0.03 | 2.1±0.2 |

### Table 3: Shock Analysis

| Event | Model | Pre-AUPRC | Event-AUPRC | Degradation |
|-------|-------|-----------|-------------|-------------|
| Terra | GraphPFN | 0.82 | 0.74 | -0.08 |
| Terra | ROLAND | 0.72 | 0.61 | -0.11 |
| FTX | GraphPFN | 0.81 | 0.76 | -0.05 |
| FTX | ROLAND | 0.71 | 0.63 | -0.08 |

### Table 4: Robustness to Negative Sampling

| Train neg:pos | Test neg:pos | Scheme | AUPRC ↑ | AUROC ↑ |
|---------------|--------------|--------|---------|---------|
| 5:1 | 5:1 | uniform | [TBD] | [TBD] |
| 5:1 | 10:1 | uniform | [TBD] | [TBD] |
| 5:1 | 50:1 | uniform | [TBD] | [TBD] |
| 5:1 | 5:1 | degree-biased | [TBD] | [TBD] |

### Table 5: Imputation Results

| Mask Rate | Edge Recall ↑ | Edge MAE ↓ | Node MAE ↓ |
|-----------|---------------|------------|------------|
| 10% | [TBD] | [TBD] | [TBD] |
| 20% | [TBD] | [TBD] | [TBD] |
| 30% | [TBD] | [TBD] | [TBD] |

### Figure List

1. **Fig 1**: Multi-step forecasting performance comparison (bar chart)
2. **Fig 2**: Rolling forecasting AUPRC over time (line plot)
3. **Fig 3**: Precision-Recall curve on representative test window
4. **Fig 4**: Network statistics evolution (Gini, HHI, Density)
5. **Fig 5**: Shock analysis - TVL and edge changes around Terra/FTX
6. **Fig 6**: Edge weight error distribution (histogram)
7. **Fig 7**: Hyperparameter sensitivity analysis

---

## 13. 附录: 代码结构

```
graphpfn/
├── run_full_experiment.py      # 唯一主实验脚本 (统一入口)
│   ├── strict_temporal_split()     # 严格时间划分 (2020-2024/2025)
│   ├── compute_data_quality()      # 数据质量统计
│   ├── save_predictions_csv()      # 预测结果输出
│   ├── compute_recall_at_k()       # Recall@K 指标
│   ├── compute_weighted_mae()      # Weighted MAE 指标
│   ├── run_graphpfn_experiment()   # Task I: GraphPFN (Frozen/Finetuned)
│   ├── run_roland_experiment()     # Task I: ROLAND baseline
│   ├── run_shock_analysis()        # Task II: Shock Analysis
│   ├── run_imputation_experiment() # Task III: Imputation
│   └── run_network_statistics()    # Network metrics
├── run_optuna_search.py        # Hyperparameter optimization (可选)
├── src/
│   └── network_statistics.py   # Network metrics module
├── data/
│   ├── historical-network_week_2025-07-01.json
│   └── meta_df.csv
├── checkpoints/
│   └── graphpfn-v1.ckpt        # Pretrained GraphPFN
└── output/
    ├── experiment_results.json     # 完整实验结果
    ├── data_quality.json           # 数据质量统计
    ├── predictions_edges_*.csv     # 边级预测
    └── predictions_nodes_*.csv     # 节点级预测
```

---

## 14. Appendix: 超参数详表

| Component | Setting |
|-----------|---------|
| Time split | Rolling walk-forward (2020-2024) + Hold-out (2025) |
| Encoder | GraphPFN v1 (graphpfn-v1.ckpt) |
| Link scorer | MLP([h_u, h_v, h_u⊙h_v, \|h_u-h_v\|]) |
| Loss weights | λ₁=1.0 (exist), λ₂=1.0 (weight), λ₃=0.5 (node) |
| Optimizer | Adam |
| LR | 1e-3 (default), [1e-5, 1e-2] (Optuna) |
| Weight decay | 1e-4 (default), [1e-6, 1e-3] (Optuna) |
| Early stopping | Val AUPRC, patience=5 |
| Neg sampling | uniform, neg:pos=5:1, seed=42 |

---

## 15. 参考文献

1. GraphPFN: Hollmann et al., "Graph-Tabular Prior-Data Fitted Networks" (2024)
2. ROLAND: You et al., "ROLAND: Graph Learning Framework for Dynamic Graphs" (2022)
3. EvolveGCN: Pareja et al., "EvolveGCN: Evolving Graph Convolutional Networks" (2020)
4. DySAT: Sankar et al., "DySAT: Deep Neural Representation Learning on Dynamic Graphs" (2020)

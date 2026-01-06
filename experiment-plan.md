# DeXposure × GraphPFN（全量）实验计划
> **主线叙事NBER：**  
> 我们把 DeFi 系统视为“时序信用敞口网络”。**机制层**是协议间敞口边（edge exposures）的形成与演化；**状态层**是协议规模（TVL proxy）变化。  
> 因此以 **edge-level exposure forecasting（exist + weight）为主任务**，以 **node-level TVL log-change 为辅助任务**进行多任务训练。  
> 核心对照：**Frozen-encoder probing vs DeXposure-finetuned GraphPFN**，衡量“微调带来的经济测度/预测增益”。

---

## 0. 研究目标与交付物

### 0.1 目标
在 DeXposure 全量周频数据上，用 GraphPFN 作为 temporal-graph-tabular backbone，训练一个可解释的 exposure forecasting foundation model，并证明微调显著提升（以及可选地对齐/超过 repo 强基线）。

### 0.2 最终交付物（可复现）
- `metrics.json`：各模型在 Test（及 rolling 汇总）的 AUPRC/AUROC、MAE/RMSE、Recall@K 等
- `predictions_edges_test.csv`：边级预测（存在性 + 边权）
- `predictions_nodes_test.csv`：节点 TVL log-change 预测
- `data_quality.json`：每周数据质量与缺失统计
- （论文）Table 1–5 + Figure 1–7（下文提供占位模板）

---

## 1. 数据与全量读取（必须流式）

### 1.1 数据源
- 全量：`historical-network_week_2020-03-30.json`（1.1GB，按周快照）
- 元数据：`meta_df.csv`（`id,name,category`）
- （可选事件研究）`network_data/filtered_*_terra.csv`, `filtered_*_ftx.csv`

### 1.2 JSON 快照结构（每周）
每个周 `YYYY-MM-DD` 包含：
- `nodes`: `{id, size, composition}`
- `links`: `{source, target, size, composition}`，其中 `target` 可能为 `null`（外部资产）

### 1.3 清洗规则（主实验）
- 丢弃 `target == null` 的边（并记录比例）
- 若 `source/target` 不在当周节点集合，丢弃并记录比例
- `category` join 不到 → 置 `"Unknown"`

### 1.4 数据质量统计（写入 data_quality.json）
每周记录：
- `N_nodes`, `N_edges`
- `pct_target_null_dropped`
- `pct_endpoint_missing_dropped`
- `overlap_ratio_next_week = |V_t ∩ V_{t+1}| / |V_t|`
- `pct_category_unknown`

---

## 2. 任务结构：主边辅点（机制层 + 状态层）

给定周 t 的图 \(G_t=(V_t,E_t)\) 与特征，预测周 t+1。

### 2.1 主任务A：Edge existence（分类）
\[
y^{exist}_{t+1}(u,v)=\mathbb{1}[(u,v)\in E_{t+1}]
\]

### 2.2 主任务B：Edge weight（回归，仅正例边）
\[
y^{w}_{t+1}(u,v)=\log(1+\text{size}_{t+1}(u,v))
\]

### 2.3 辅助任务：Node TVL log-change（回归）
\[
y^{node}_{t+1}(u)=\log(1+\text{size}_{t+1}(u))-\log(1+\text{size}_{t}(u))
\]
仅对 \(V_t \cap V_{t+1}\) 计算标签。

---

## 3. 特征工程（Temporal-Graph-Tabular）

### 3.1 节点特征 x_t(u)（MVP 默认）
- `log_size = log1p(size_t(u))`
- `num_tokens = len(composition)`
- `max_share = max(comp.values)/(size+eps)`（size=0或comp空→0）
- `entropy(comp) = -Σ p_k log(p_k+eps)`，`p_k=value_k/(sum(values)+eps)`
- `category`（one-hot 或 embedding；默认 one-hot）

> eps 建议 `1e-9`

### 3.2 边特征（可选保留）
- `edge_log_weight = log1p(edge_size_t(u,v))`
> 第一版可不喂给 scorer，只保留用于后续增强/解释。

---

## 4. 模型：路线1（Node Encoder + Link Scorer）

### 4.1 Node Encoder（GraphPFN）
输入：当周图 `edge_index` + 节点特征 `X_t`  
输出：节点嵌入 \(h_u \in \mathbb{R}^d\)

### 4.2 Link Scorer（两头：exist + weight）
对任意 pair (u,v) 构造：
- \(z = [h_u, h_v, h_u \odot h_v, |h_u-h_v|]\)

输出：
- exist：\(p_{uv} = \sigma(\text{MLP}_{exist}(z))\)
- weight：\(\hat w_{uv} = \text{MLP}_{weight}(z)\)（回归 log1p 权重）

### 4.3 Node Head（辅助任务）
- \(\hat y^{node}_u = \text{MLP}_{node}(h_u)\)（回归 ΔlogTVL）

---

## 5. 训练样本构造（周对）+ 负采样（默认）

### 5.1 周对样本
样本单位：每个周对 `(t -> t+1)`  
训练输入用 \(G_t\)，标签来自 \(t+1\)。

### 5.2 正例边 pos
- `pos_edges = E_{t+1}`（已清洗）

### 5.3 负采样 neg（默认：uniform，neg:pos=5:1）
从 `V_t × V_t`（有向）随机采样 (u,v)，满足：
- (u,v) 不在 `pos_edges`
- 采样数：`|neg| = 5 * |pos|`
- 固定随机种子，保证可复现

> 稳健性：可在附录加入 degree-biased negatives 或更高 test neg 比例。

### 5.4 pairs 列表（用于存在性训练）
- `pairs = pos_edges ∪ neg_edges`
- `label_exist ∈ {0,1}`
- `label_weight = log1p(size_{t+1}(u,v))`（仅正例；负例 weight mask=False）

### 5.5 node 标签与 mask（辅助任务）
对共同节点 `V_t ∩ V_{t+1}`：
- `y_node(u) = log1p(size_{t+1}) - log1p(size_t)`
- `node_mask` 标识哪些节点有标签

---

## 6. 损失函数（多任务）

### 6.1 Exist loss（分类）
- `L_exist = BCEWithLogitsLoss`（pos+neg 全部 pairs）

### 6.2 Weight loss（回归，仅正例）
- `L_weight = SmoothL1 / MAE`（pos pairs only）

### 6.3 Node loss（辅助回归）
- `L_node = SmoothL1 / MAE`（node_mask=True）

### 6.4 总损失（默认权重）
\[
\mathcal{L}=1.0\cdot L_{exist}+1.0\cdot L_{weight}+0.5\cdot L_{node}
\]
> Ablation：设置 λ3=0（移除 node aux）看主任务是否下降。

---

## 7. 训练与评估协议（全量）

### 7.1 时间切分（两种）
**Option A（先跑通）**：固定比例按时间
- Train 70% / Val 15% / Test 15%

**Option B（更经济学）**：rolling walk-forward（推荐最终）
- 初始训练窗口：52或104周
- 每 4 周滚动一次（train→test）
- 汇总 test 指标 mean±std/CI

### 7.2 核心对照：Frozen-encoder probing vs Finetuned
> 说明：严格 zero-shot（encoder+scorer 都不训练）通常不可用，因为 scorer 没训练难以预测。  
> 论文里建议采用金融/ML都常用的 **frozen probing** 作为“未微调对照”。

- **Frozen-encoder probing**
  - GraphPFN encoder **冻结**
  - 仅训练 link scorer（+ 可选 node head）
  - 解释：衡量预训练表示的可迁移性（representation quality）

- **Finetuned GraphPFN**
  - encoder + scorer（+ node head）一起训练
  - 解释：衡量 DeXposure 微调增益（finetuning gain）

（可选外部强基线）
- RO-LAND（DeXposure repo）用于 benchmark 对照

---

## 8. 指标（统一口径）

### 8.1 Edge existence（稀疏）
- **AUPRC**（主）
- AUROC（辅）
> 评估时明确负例集合：固定 neg:pos 与 seed；可加 “更高 neg 比例” stress test。

### 8.2 Edge weight（仅正例边）
- MAE / RMSE（log1p 空间）

### 8.3 Tail / 系统性重要性
- Recall@K（K=100/500/1000 或 top 0.1%/1%）
- Weighted MAE（按真实边权加权）

### 8.4 Node ΔlogTVL（辅助）
- MAE / RMSE（mask=True）

---

## 9. 输出文件规范（便于论文与复现实验）

### 9.1 predictions_edges_test.csv
- `time_t, time_t1, u_id, v_id, y_exist_true, y_exist_pred, y_w_true, y_w_pred, is_positive`

### 9.2 predictions_nodes_test.csv
- `time_t, time_t1, node_id, y_node_true, y_node_pred, size_t, category`

### 9.3 metrics.json
- split/rolling 汇总：AUPRC/AUROC、MAE/RMSE、Recall@K、Weighted MAE
- 并列保存 Frozen vs Finetuned（+ 可选 RO-LAND）

### 9.4 data_quality.json
- 见 1.4

---

## 10. 经济学解释层（论文必须写的“结构化思维”）

### 10.1 Measurement（测度层）
- edge size：解释为跨协议资产依赖/信用敞口强度（风险传染通道）
- node size：TVL proxy（规模状态）

### 10.2 Derived risk metrics（派生风险层）
从预测网络 \(\hat{G}_{t+1}\) 推导：
- out-/in-exposure
- 最大单一对手敞口占比（max counterparty share）
- 敞口集中度（Herfindahl）
- 系统重要性排名（Top-10/Top-50）

### 10.3 Policy/Counterfactual（反事实/压力测试，NBER风格）
- 对某类 category 节点施加 shock（size 下调 30%）
- 或对某类边施加 shrink/cut
- 比较预测风险指标的变化 Δ（就是政策分析形式）

### 10.4 Imputation（缺失修复，可选但强贴合 NBER call）
- 随机 mask 10%/30% edges（或按缺失模式）
- 评估模型对被 mask 子集的 exist/weight 重建能力

---

# 论文图表占位模板（直接粘贴到论文）

## Tables

### Table 1. Main results: Frozen-encoder probing vs Finetuned GraphPFN (and optional RO-LAND)
| Model | Encoder | Scorer | Edge Exist AUPRC ↑ | Edge Exist AUROC ↑ | Edge Weight MAE ↓ (log1p, pos-only) | Edge Weight RMSE ↓ (log1p, pos-only) | Tail Recall@K ↑ (K=100/500/1000) | Node ΔlogTVL MAE ↓ (aux) |
|---|---|---|---:|---:|---:|---:|---:|---:|
| GraphPFN (Frozen Probing) | frozen | trained | [TBD] | [TBD] | [TBD] | [TBD] | [TBD] | [TBD] |
| **GraphPFN (Finetuned)** | finetuned | trained | **0.814** | **0.958** | **3.23** | **4.05** | [TBD] | **0.125** |
| RO-LAND (DeXposure repo) *(optional)* | n/a | n/a | [TBD] | [TBD] | [n/a] | [n/a] | [TBD] | [n/a] |

> **实验结果说明** (2026-01-06):
> - 使用 `run_dexposure_experiment.py` 运行 Link Prediction 实验
> - 配置: neg_ratio=5, train_ratio=0.7, val_ratio=0.15, hidden_dim=256, epochs=5, seed=42
> - 详细结果见 `output/dexposure_graphpfn_link/metrics.json`

---

### Table 2. Task formulation ablation: why two-head (exist + weight) matters
| Setting | Edge Exist AUPRC ↑ | Edge Exist AUROC ↑ | Edge Weight MAE ↓ (log1p, pos-only) | Tail Recall@K ↑ | Notes |
|---|---:|---:|---:|---:|---|
| Exist-only | [TBD] | [TBD] | [n/a] | [TBD] | Predict only existence |
| Exist + Weight (two-head) | [TBD] | [TBD] | [TBD] | [TBD] | Default main setting |
| Weight-only (treat non-edge as 0) | [TBD] | [TBD] | [TBD] | [TBD] | Often unstable under sparsity |

---

### Table 3. Model ablation: mechanism vs state, and what features matter
| Ablation (Finetuned unless noted) | Remove Graph? | Remove Category? | Remove Composition Summary? | Remove Node Aux Task? | Edge Exist AUPRC ↑ | Edge Weight MAE ↓ | Tail Recall@K ↑ |
|---|---:|---:|---:|---:|---:|---:|---:|
| Default (Finetuned) | No | No | No | No | [TBD] | [TBD] | [TBD] |
| No-Graph (tabular-only) | Yes | No | No | No | [TBD] | [TBD] | [TBD] |
| No-Category | No | Yes | No | No | [TBD] | [TBD] | [TBD] |
| No-Composition Summary | No | No | Yes | No | [TBD] | [TBD] | [TBD] |
| No-Node-Aux (λ3=0) | No | No | No | Yes | [TBD] | [TBD] | [TBD] |
| Frozen Probing (reference) | No | No | No | No | [TBD] | [TBD] | [TBD] |

---

### Table 4. Robustness to negative sampling and class imbalance
| Neg Sampling Setup | Train neg:pos | Test neg:pos | Neg Scheme | Edge Exist AUPRC ↑ | Edge Exist AUROC ↑ | Notes |
|---|---:|---:|---|---:|---:|---|
| Default | 5:1 | 5:1 | uniform | [TBD] | [TBD] | Main reported setting |
| More realistic imbalance | 5:1 | 10:1 | uniform | [TBD] | [TBD] | Harder test |
| Even harder test | 5:1 | 50:1 | uniform | [TBD] | [TBD] | Stress evaluation |
| Degree-biased negatives | 5:1 | 5:1 | degree-biased | [TBD] | [TBD] | Robustness check |

---

### Table 5. Imputation (optional, aligned with economic measurement)
| Mask Rate | Mask Target | Metric (Exist) AUPRC ↑ | Metric (Weight) MAE ↓ | Frozen Probing | Finetuned | Notes |
|---:|---|---:|---:|---:|---:|---|
| 10% | edges | [TBD] | [TBD] | [TBD] | [TBD] | random mask |
| 30% | edges | [TBD] | [TBD] | [TBD] | [TBD] | random mask |
| 10% | nodes *(optional)* | [TBD] | [TBD] | [TBD] | [TBD] | if node masking implemented |

---

## Figures

### Figure 1. Rolling forecasting performance over time (Edge existence AUPRC)
**[INSERT FIGURE 1 HERE]**

### Figure 2. Precision–Recall curve on a representative test window (or crisis window)
**[INSERT FIGURE 2 HERE]**

### Figure 3. Tail risk relevance: Recall@K for future largest exposures
**[INSERT FIGURE 3 HERE]**

### Figure 4. Edge weight error distribution (positive edges only)
**[INSERT FIGURE 4 HERE]**

### Figure 5. Economic measurement: derived systemic risk metrics (true vs predicted networks)
**[INSERT FIGURE 5 HERE]**

### Figure 6. Sector-to-sector exposure matrix heatmaps (true vs predicted)
**[INSERT FIGURE 6 HERE]**

### Figure 7. Event study around crises (optional): Terra / FTX windows
**[INSERT FIGURE 7 HERE]**

---

## Appendix (可选占位)

### Appendix Table A1. Hyperparameters and training protocol
| Component | Setting |
|---|---|
| Time split | [fixed 70/15/15 OR rolling walk-forward] |
| Encoder | GraphPFN [version/hash] |
| Link scorer | MLP([h_u,h_v,h_u*h_v,|h_u-h_v|]) |
| Loss weights | λ1=1.0 (exist), λ2=1.0 (weight), λ3=0.5 (node) |
| Optimizer | AdamW |
| LR / WD | [TBD] |
| Early stopping | Val AUPRC / MAE, patience=[TBD] |
| Neg sampling | uniform, neg:pos=5:1, seed=[TBD] |

### Appendix Table A2. Data quality summary
| Statistic | Mean | Std | Min | Max |
|---|---:|---:|---:|---:|
| nodes per week | [TBD] | [TBD] | [TBD] | [TBD] |
| edges per week | [TBD] | [TBD] | [TBD] | [TBD] |
| pct target=null dropped | [TBD] | [TBD] | [TBD] | [TBD] |
| overlap ratio (t,t+1) | [TBD] | [TBD] | [TBD] | [TBD] |
| pct category=Unknown | [TBD] | [TBD] | [TBD] | [TBD] |
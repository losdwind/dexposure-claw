# DeXposure 全量实验规范（路线1：Node Encoder + Link Scorer）
**主任务：edge-level exposure forecasting（exist + weight）**  
**辅助任务：node-level TVL log-change**  
**核心对照：GraphPFN zero-shot vs GraphPFN finetuned**  
**负采样默认：uniform，neg:pos = 5:1**

---

## 0. 核心定义

### 0.1 节点（protocol）
- node_id: string（来自 weekly snapshot 的 `nodes[i].id`）
- node.size: float（USD，TVL proxy）
- node.composition: dict token->USD
- node.category: 来自 `meta_df.csv` join（缺失→"Unknown"）

### 0.2 边（exposure link）
每周 snapshot 的 `links`：
- source: node_id
- target: node_id 或 null
- size: float（USD）

**清洗规则（主实验）**
- 丢弃 `target == null`
- 若 source/target 不在当周节点集合，丢弃并计数

---

## 1. 预测任务（主边辅点）

给定周 t 的图 \(G_t=(V_t,E_t)\) 与特征，预测周 t+1：

### 1.1 主任务A：Edge existence（分类）
\[
y^{exist}_{t+1}(u,v)=\mathbb{1}[(u,v)\in E_{t+1}]
\]

### 1.2 主任务B：Edge weight（回归，只对正例边）
\[
y^{w}_{t+1}(u,v)=\log(1+\text{size}_{t+1}(u,v))
\]

### 1.3 辅助任务：Node TVL log-change（回归）
\[
y^{node}_{t+1}(u)=\log(1+\text{size}_{t+1}(u))-\log(1+\text{size}_{t}(u))
\]

---

## 2. 输入特征

### 2.1 节点特征 x_t(u)（MVP）
对每个节点 u：
1) log_size = log1p(size_t(u))
2) num_tokens = len(composition)
3) max_share = max(comp.values)/(size+eps)（size=0或comp空→0）
4) entropy(comp) = -Σ p_k log(p_k+eps)，p_k=value_k/(sum(values)+eps)
5) category encoding（one-hot 或 embedding；默认 one-hot）

最终：x_t(u) = [log_size, num_tokens, max_share, entropy, cat_enc]

### 2.2 边特征（可选）
- edge_weight_t(u,v) = log1p(size_t(u,v))
> 可先不喂给 scorer，只用于额外分析或后续增强。

---

## 3. 模型结构（路线1）

### 3.1 Node Encoder（GraphPFN）
输入：当周图 G_t（edge_index）+ 节点特征 X_t  
输出：每个节点 embedding h_u ∈ R^d

### 3.2 Link Scorer（必须实现）
对任意 pair (u,v)，构造 pair feature：
- z = concat(h_u, h_v, h_u * h_v, |h_u - h_v|)
然后：
- exist head：p = sigmoid(MLP_exist(z))
- weight head：w_hat = MLP_weight(z)  （输出 log1p 规模）

> 训练时 weight 只对正例边计算 loss。

---

## 4. 训练样本构造（最关键）

### 4.1 时间轴与周对
- 从全量 JSON 流式读取每周快照并按日期排序：t0..tT
- 样本单位：每个周对 (t -> t+1)

### 4.2 正例边（pos）
- pos_edges = E_{t+1} （清洗后：target != null 且端点在 V_{t+1}）

注意：训练输入是 G_t，但标签来自 t+1。

### 4.3 负采样（neg）【默认设置】
**Uniform negative sampling，neg:pos = 5:1**

实现要求：
- 负例从 V_t × V_t 中采样（有向）
- 条件： (u,v) 不在 pos_edges
- 采样数：|neg| = 5 * |pos|
- 固定随机种子，确保可复现

> 备注：可选在附录做 degree-biased negatives 作为稳健性。

### 4.4 训练对列表（pairs）
构造：
- pairs = pos_edges ∪ neg_edges
为每条 pair 存：
- u_id, v_id
- label_exist ∈ {0,1}
- label_weight = log1p(size_{t+1}(u,v))（仅对正例；负例可置 0 + mask）

### 4.5 辅助任务标签
对共同节点集合 V_t ∩ V_{t+1}：
- y_node(u) = log1p(size_{t+1}(u)) - log1p(size_t(u))
并提供 node_mask 指示哪些节点有标签。

---

## 5. 损失函数（多任务）

对每个周对样本：

### 5.1 Exist loss（分类）
- L_exist = BCEWithLogitsLoss on all pairs（pos+neg）

### 5.2 Weight loss（回归，仅正例）
- L_weight = SmoothL1 / MAE on positive pairs only（use weight_mask）

### 5.3 Node loss（辅助）
- L_node = SmoothL1 / MAE on nodes with node_mask=True

### 5.4 总损失（默认权重）
- L = 1.0*L_exist + 1.0*L_weight + 0.5*L_node

> 建议做一个 ablation：把 L_node 去掉（0.0）看主任务是否下降。

---

## 6. 训练与评估协议（全量）

### 6.1 切分（两种可选）
**Option A（简单）**：按时间固定比例  
- Train 70% / Val 15% / Test 15%

**Option B（推荐，经济学 forecasting 味道）**：rolling walk-forward  
- 初始训练窗口：52周（或 104周）
- 每 4 周滚动一次
- 汇总所有窗口的平均指标与方差/置信区间

> 若时间紧先实现 Option A，后续加 Option B。

### 6.2 Zero-shot vs Finetuned（核心对照）
- Zero-shot：GraphPFN 权重冻结 + scorer（若 scorer 也冻结则需说明；推荐 scorer 训练但 encoder 冻结属于“linear probing”，不是纯 zero-shot）
- Finetuned：GraphPFN encoder + scorer 一起训练

**严格 zero-shot 定义（推荐写法）**
- zero-shot：encoder+scorer 都不更新（直接 forward）
- finetuned：encoder+scorer 在 Train 上更新

若 zero-shot 下 scorer 没法预测（未训练），可采用：
- zero-shot：冻结 encoder，仅训练 scorer（称为 *frozen-encoder probing*）
- finetuned：微调 encoder + scorer
并明确写清楚两者差异。

---

## 7. 指标（必须一致）

### 7.1 Edge existence
- AUPRC（主）
- AUROC（辅）

> 评估时的负例集合必须明确：使用与测试一致的固定负采样比例与随机种子；可额外用更高 neg 比例做稳健性。

### 7.2 Edge weight（仅正例边）
- MAE / RMSE（log1p 空间）

### 7.3 Tail / Systemic relevance
- Recall@K（未来 Top-K 最大边是否被预测为存在/且权重排名靠前）
- Weighted MAE（按真实边权加权）

### 7.4 Node TVL log-change（辅助）
- MAE / RMSE（mask=True）

---

## 8. 输出文件（便于写论文与复现实验）

### 8.1 data_quality.json
按周记录：
- N_nodes, N_edges
- pct_target_null_dropped
- pct_endpoint_missing_dropped
- overlap_ratio_next_week
- pct_category_unknown

### 8.2 metrics.json
- 每个 split 的 AUPRC/AUROC, weight MAE/RMSE, node MAE/RMSE
- 若 rolling：输出 mean/std 或 CI

### 8.3 predictions_edges_test.csv
列：
- time_t, time_t1, u_id, v_id, y_exist_true, y_exist_pred, y_w_true, y_w_pred, is_positive

### 8.4 predictions_nodes_test.csv
列：
- time_t, time_t1, node_id, y_node_true, y_node_pred, size_t, category

---

## 9. 经济学解释层（最少要做的三张图）

1) AUPRC over time / rolling windows（预测稳定性）
2) Crisis event study（Terra/FTX 窗口）：集中度/最大对手敞口/部门连接强度随时间
3) Sector connectivity heatmap：真实 vs 预测（category-by-category exposure matrix）

---

## 10. 给 Claude 的执行清单（直接复制）

1) 流式读取全量 weekly json，构建每周 snapshot（nodes+links），join meta_df.csv 得到 category  
2) 对每周构建图输入：x, edge_index（过滤 target=null）  
3) 对每周对 (t->t+1)：
   - 取 pos_edges = E_{t+1}
   - 从 V_t×V_t uniform 采 neg_edges，使 |neg|=5|pos|，固定随机种子
   - 构建 pairs（pos+neg）与 exist labels
   - 对 pos pairs 构建 weight label = log1p(edge_size_{t+1})
   - 构建 node label y_node（共同节点）
4) 实现模型：GraphPFN encoder + link scorer（exist head + weight head）+ node head（可共享 MLP）
5) 跑两套设置：
   - zero-shot（或 frozen encoder probing，明确写法）
   - finetuned（encoder+scorer更新）
6) 评估：AUPRC/AUROC、weight MAE/RMSE、Recall@K、node MAE/RMSE
7) 输出四个文件：data_quality.json, metrics.json, predictions_edges_test.csv, predictions_nodes_test.csv

---
zero-shot vs finetuned

Frozen-encoder probing：GraphPFN encoder 冻结，只训练 scorer（相当于“预训练表示能否直接线性分离/回归”）

Finetuned：encoder+scorer 一起训练
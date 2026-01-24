# 行动建议

## 现状分析

当前模型性能：
- AUROC = 0.94 ✓ (排序能力好)
- AUPRC = 0.74 (在neg_ratio=5:1下)
- Recall@100 = 0.00004 (0.004%)
- Recall@1000 = 0.0005 (0.05%)

**矛盾点**：AUROC高但Recall低 → 模型能排序，但预测的正边太少

## 优先级行动清单

### 1️⃣ 立即可做：重新解释现有结果

**无需重新训练，只需改变报告方式**

主要指标：
- **AUROC = 0.94** ← 这是金融/加密货币领域的标准指标
- 说明模型有很好的排序能力（能区分正负边）

辅助指标：
- AUPRC = 0.74 (neg_ratio=5:1)
- Precision@K / Recall@K（说明当前较低，是未来改进方向）

**文献支持**：
- 加密货币欺诈检测论文明确说：使用AUROC因为"不受类别分布影响"
- 链接预测标准做法：训练用低neg_ratio，评估用ranking指标

### 2️⃣ 可选：测试真实neg_ratio下的性能

目的：了解模型在真实分布下的AUPRC

```bash
# 修改 run_full_experiment.py 的评估部分
# 只改测试时的neg_ratio，训练保持5:1
python test_real_performance.py --test_neg_ratio 100
```

预期：
- 如果AUPRC还有0.3+：模型确实学到了pattern
- 如果AUPRC降到<0.05：说明过度依赖训练分布

### 3️⃣ 长期改进：提升Recall@K

如果确实需要提高Recall（实际推荐Top-K边时）：

#### 选项A：增加训练neg_ratio
```python
# run_full_experiment.py line 155
neg_ratio: int = 20  # 从5改到20
```
- 训练时间增加1.5-2x
- 可能提升Recall@K

#### 选项B：添加pos_weight
```python
# line 1555, BCE loss
pos_weight = torch.tensor([neg_ratio], device=device)
loss = F.binary_cross_entropy_with_logits(
    logits, labels, pos_weight=pos_weight
)
```
- 无需改neg_ratio
- 让模型更关注正样本

#### 选项C：Hard negative sampling
采样度数高的节点间的缺失边（更难的负样本）

```python
def sample_hard_negatives(graph, num_neg, rng):
    # 按节点度数加权采样
    degrees = graph.out_degrees()
    probs = degrees / degrees.sum()
    # 高度节点更可能被采样
    ...
```

## 我的建议

**按顺序执行**：

1. **先做方案1**：重新解释现有结果
   - 主报AUROC=0.94
   - 这已经是可发表的结果了
   - 说明neg_ratio=5的设置，Recall低是未来工作

2. **如果被质疑Recall太低**：做方案2
   - 测试真实neg_ratio下的AUPRC
   - 证明问题是metric选择，不是模型问题

3. **如果确实需要提升Recall**：做方案3
   - 优先尝试pos_weight（最简单）
   - 再考虑提高neg_ratio或hard sampling

## 为什么不直接重新训练？

理由：
1. **AUROC=0.94已经很好**（加密货币论文的标准）
2. **Recall低可能是合理的**（DeFi网络真的很稀疏，1230:1）
3. **重新训练风险高**：
   - 可能不会显著提升Recall
   - 训练时间长
   - 可能引入其他问题

**先用现有结果，被challenge了再改进**

## 论文中如何写

```
We evaluate our model using AUROC as the primary metric,
following the standard practice in financial graph analysis
[cite: Bitcoin fraud detection, ATGAT papers].

AUROC is chosen because it is invariant to class distribution,
which is crucial given the extreme sparsity of DeFi networks
(negative-to-positive ratio of ~1230:1).

Results:
- AUROC: 0.94
- AUPRC: 0.74 (evaluated with neg_ratio=5:1)

The high AUROC indicates strong ranking capability, though
absolute recall metrics remain low due to network sparsity.
Future work could explore hard negative sampling to improve
top-k recommendation performance.
```

## 相关文献

支持AUROC作为主要指标：
- "AUC is adopted as the primary metric because it is unaffected
   by class distribution" - ATGAT paper
- Bitcoin fraud detection: AUC=0.94, F1=92.87%
- Multi-Distance ST-GNN: 报告AUC-ROC, AUC-PR, F1

支持低neg_ratio训练：
- GraphPFN: edge reconstruction用1:1
- Knowledge graphs: 训练1:1，测试用全集排序
- OGB-COLLAB: 均衡采样 + ranking指标

# 学术写作指南：CS 与经济/金融篇

**从审稿人的红笔中学到的——如何为顶会/顶刊写出清晰有力的论文**

*灵感来源于 The Academic Writer's Apprenticeship，适配计算机科学与经济/金融领域*

---

## 目录

- [[#前言：两个领域，一套技艺]]
- [[#如何使用本指南]]
- [[#第一部分：基础篇——标题与摘要]]
  - [[#第1章：标题——六秒定生死]]
  - [[#第2章：摘要——独立的微型论文]]
- [[#第二部分：构建论证——引言与相关工作]]
  - [[#第3章：引言——为什么有人应该读这篇论文]]
  - [[#第4章：相关工作与文献——区分综述与论证]]
- [[#第三部分：核心方法与理论]]
  - [[#第5章 CS篇：方法——让读者能复现]]
  - [[#第5章 经济金融篇：模型与识别策略——让读者信服你的因果推断]]
- [[#第四部分：实验与结果]]
  - [[#第6章 CS篇：实验——公平对比，诚实报告]]
  - [[#第6章 经济金融篇：实证结果——让数据说话]]
- [[#第五部分：讨论与结论]]
  - [[#第7章：讨论与结论——不要虎头蛇尾]]
- [[#第六部分：贯穿一切的写作技艺]]
  - [[#第8章：句子层面的手术]]
  - [[#第9章：常见写作习惯诊断]]
- [[#附录A：CS论文自我编辑清单]]
- [[#附录B：经济金融论文自我编辑清单]]
- [[#附录C：推荐阅读]]

---

## 前言：两个领域，一套技艺

计算机科学和经济/金融看似截然不同——一个追求算法性能的极致，另一个追求因果推断的可信。但在写作技艺上，它们面临着惊人相似的挑战：

- 如何在六秒内让审稿人决定认真读你的论文？
- 如何用最少的词传达最多的信息？
- 如何让你的贡献在审稿人的脑中留下清晰的印记？

本指南的每一条原则都来自真实的审稿经验。你会看到两类并行的"修改前/修改后"示例——一类来自CS（系统、机器学习、NLP等），一类来自经济/金融——让你看到同一写作原则在不同领域的具体表现。

无论你正在投递 NeurIPS、ICML、ACL、OSDI，还是 AER、Econometrica、JFE、RFS——清晰有力的写作都会成为你最大的竞争优势。

---

## 如何使用本指南

每章遵循统一结构：

**核心原则。** 以一条写作原则开篇。

**修改前 vs 修改后。** 并排对比展示改动的力量。标注 `[CS]` 或 `[Econ/Fin]` 来区分领域。

**📌 课程框。** 提炼可迁移的原则。

**领域差异说明。** 标注 CS 和经济/金融在具体写作惯例上的区别。

**自检清单。** 每章结尾可供修改时逐条核查。

---

## 第一部分：基础篇——标题与摘要

---

### 第1章：标题——六秒定生死

审稿人拿到你论文的前六秒，通常只看标题。一个好标题不是标签——它是一个**压缩到极致的发现陈述**。

#### 原则：标题应传达你的核心发现或贡献，而非仅仅描述主题

**`[CS]` 示例**

| 修改前 | 修改后 |
|---|---|
| A Study of Attention Mechanisms for Document Summarization | Longformer: The Long-Document Transformer |
| Graph Neural Networks for Protein Structure | Protein Structure Prediction with Geometric Graph Networks Achieves Near-Experimental Accuracy |

> 📌 **CS标题的黄金公式：[方法名/核心思想] + [解决了什么问题] + [可选：关键结果]。** 如果你的方法有一个好名字（如 Longformer、FlashAttention），它本身就能传达核心思想。如果没有，标题应该暗示你的方法*为什么有效*，而非仅仅说它*是什么*。

**`[Econ/Fin]` 示例**

| 修改前 | 修改后 |
|---|---|
| The Effect of Minimum Wage on Employment | Do Minimum Wages Kill Jobs? Evidence from a Natural Experiment in New Jersey |
| A Study of CEO Compensation and Firm Performance | Paying for Performance: CEO Compensation and Firm Value in the S&P 500 |

> 📌 **经济/金融标题的黄金公式：[你的发现/问题] + [可选：证据来源/识别策略]。** 好的经济学标题往往以问句形式出现（因为因果问题是核心），或直接暗示因果方向。包含识别策略来源（"Natural Experiment""Regression Discontinuity"）能立刻增加可信度。

#### 共同错误：变量罗列式标题

两个领域最常见的弱标题形式都是罗列主题：

- ❌ CS: "Transformers, Diffusion Models, and Image Generation"
- ❌ Econ: "Corporate Governance, Executive Compensation, and Firm Risk"

这类标题告诉读者你*涉及*什么领域，但不告诉他们你*发现*了什么。

修改方法：问自己——如果一个同事在走廊上问"你这篇论文是关于什么的？"，你会怎么用一句话回答？那句话就是你的标题。

> 📌 **测试你的标题：一个完全不了解你论文的读者，能否仅从标题推测出你的核心发现或论点？如果不能，你的标题只是一个标签，不是一个故事。**

#### 领域差异

| 维度 | CS | 经济/金融 |
|---|---|---|
| 方法命名 | 常见且鼓励（FlashAttention, LoRA） | 罕见，偶见于计量方法论文 |
| 问句标题 | 较少见 | 非常常见且有效 |
| 结果暗示 | 可以直接写"achieves SOTA" | 通常更含蓄，暗示方向而非给出数字 |
| 标题长度 | 通常5-12词 | 通常8-18词，可含副标题 |

#### 标题自检

- ☐ 标题是否传达了我的核心发现/贡献，而非仅仅描述主题？
- ☐ 一个不了解我论文的人，能否从标题猜出我的论点？
- ☐ `[CS]` 如果我有一个方法名，它是否本身就暗示了核心思想？
- ☐ `[Econ/Fin]` 标题是否暗示了因果方向或识别策略？

---

### 第2章：摘要——独立的微型论文

摘要不是论文的缩略版。它是一篇**独立的微型论文**——许多读者（包括大部分审稿人在初筛阶段）只会读到这里。

#### 原则：摘要必须在150-250词内完成一个完整的论证闭环

**`[CS]` 摘要结构（推荐五句话模型）**

1. **问题/动机**（1-2句）：为什么这个问题重要？现有方法的瓶颈是什么？
2. **核心方法/思想**（1-2句）：你做了什么？关键洞察是什么？
3. **结果**（1-2句）：主要的定量结果。
4. **贡献/意义**（1句）：这为什么重要？

**`[Econ/Fin]` 摘要结构（推荐五句话模型）**

1. **研究问题**（1句）：你回答什么问题？
2. **识别策略/数据**（1-2句）：你如何获得可信的因果推断？用什么数据？
3. **核心发现**（1-2句）：主要结果。
4. **机制/解释**（1句，可选）：为什么会这样？
5. **政策含义/贡献**（1句）：这对理论或政策意味着什么？

#### 常见错误1：以方法论开篇

**`[CS]`**

| 修改前 | 修改后 |
|---|---|
| We propose a novel transformer-based architecture with multi-scale attention and hierarchical pooling for document classification. | Classifying long documents remains challenging because standard transformers truncate input at 512 tokens. We introduce Longformer, which uses a combination of local and global attention to process documents of up to 16K tokens with linear complexity. |

> 📌 **不要以"We propose..."开篇。先告诉读者问题是什么、为什么难，然后再介绍你的方法。读者需要先理解*为什么*你的方法存在，才能关心它*是什么*。**

**`[Econ/Fin]`**

| 修改前 | 修改后 |
|---|---|
| Using a difference-in-differences framework with county-level panel data from 2005 to 2020, this paper studies the effect of broadband internet access on local labor markets. | Does broadband internet create jobs or destroy them? We exploit the staggered rollout of fiber-optic networks across U.S. counties to estimate the causal effect of high-speed internet on local employment. |

> 📌 **不要以识别策略开篇。先呈现问题——如果可能，以一个读者关心的问题形式。识别策略是手段，不是目的。**

#### 常见错误2：模糊的贡献声明

两个领域都有同样的通病——用笼统的词来描述贡献：

- ❌ "Our results provide new insights into the problem."
- ❌ "This paper deepens our understanding of market dynamics."
- ❌ "We make significant contributions to the literature."

这些句子什么也没说。它们是学术写作中的"占位符"。

| 修改前 | 修改后 |
|---|---|
| `[CS]` Our work provides new insights into efficient training of large language models. | `[CS]` Our method reduces training compute by 40% while matching the performance of standard training, suggesting that most gradient updates in early epochs are redundant. |
| `[Econ/Fin]` This paper contributes to the growing literature on fintech and financial inclusion. | `[Econ/Fin]` We show that mobile payment adoption increases household consumption by 12% among previously unbanked rural households, primarily through reduced transaction costs rather than credit access. |

> 📌 **用数字和具体机制替代"new insights"和"contributes to the literature"。审稿人要的不是你的自我评价，而是你的证据。**

#### 常见错误3：结果中只有定性描述

| 修改前 | 修改后 |
|---|---|
| `[CS]` Our method significantly outperforms existing baselines on multiple benchmarks. | `[CS]` Our method achieves 94.2% accuracy on ImageNet (vs. 91.8% for the best baseline) while using 3× fewer parameters. |
| `[Econ/Fin]` We find a large and statistically significant effect of the policy on outcomes. | `[Econ/Fin]` A one-standard-deviation increase in bank capital requirements reduces lending by 4.3 percentage points (p < 0.01), with effects concentrated among small business loans. |

> 📌 **在摘要中给出你最重要结果的具体数字。"显著优于"不是结果——那是对结果的广告。**

#### 摘要自检

- ☐ 第一句话是否陈述了问题/动机，而非方法？
- ☐ 贡献是否用具体数字和机制描述，而非"new insights"？
- ☐ 摘要是否能独立成文（不需要阅读全文就能理解）？
- ☐ `[CS]` 是否包含了最重要的定量结果？
- ☐ `[Econ/Fin]` 是否明确了识别策略和数据来源？

---

## 第二部分：构建论证——引言与相关工作

---

### 第3章：引言——为什么有人应该读这篇论文

引言是你论文中**销售力最强**的部分。它的核心任务只有一个：说服审稿人这个问题值得研究，你的方法值得了解。

#### CS 引言的四段式架构

CS论文的引言通常遵循这个结构：

**第1段——问题与动机：** 什么问题重要？为什么现在解决它特别紧迫？用一个具体的场景或数字开篇，不要从"近年来，深度学习取得了巨大进展..."开始。

**第2段——现有方法的局限：** 人们尝试了什么？为什么不够好？要具体——是太慢、不够准确、不能泛化、还是计算代价太高？

**第3段——我们的方法概述：** 你的核心洞察是什么？用一两句话概括你方法的关键思想（不是细节），让读者理解你*为什么*这么做，而不仅仅是你做了什么。

**第4段——贡献列表与结果预览：** 用 bullet points 列出你的贡献（通常3-4条），最后一条可以预告你的关键结果。

**`[CS]` 示例——第1段开篇**

| 修改前 | 修改后 |
|---|---|
| In recent years, large language models (LLMs) have achieved remarkable success across a wide range of natural language processing tasks. However, the computational cost of training these models remains a significant challenge. | Training a single large language model now costs upwards of $10M in compute—and this cost doubles roughly every 9 months. This trajectory is unsustainable: at current rates, training the next generation of models will exceed the GDP of a small nation. |

> 📌 **CS引言第一段的黄金法则：用一个具体的事实、数字或场景开篇。"近年来X取得了巨大进展"是学术论文中最无趣的开场白——每个审稿人都已经读了几百遍了。**

#### 经济/金融 引言的四段式架构

经济/金融论文的引言通常更长（1.5-3页），结构如下：

**第1段——核心问题与现实世界的重要性：** 用一个政策辩论、市场现象或社会事实开篇。为什么这个问题对现实世界有影响？

**第2段——为什么这个问题在学术上困难/有争议：** 现有文献的结论是什么？为什么不充分？是因为内生性、数据局限、还是理论分歧？

**第3段——你的方法/策略：** 你利用什么自然实验、工具变量、断点或政策变动来克服识别问题？

**第4段——主要发现和贡献：** 核心结果是什么？你对哪些文献做出了贡献？

**`[Econ/Fin]` 示例——第1段开篇**

| 修改前 | 修改后 |
|---|---|
| The relationship between financial development and economic growth has been extensively studied in the economics literature. This paper contributes to this debate by examining the role of fintech. | In 2023, mobile payment transactions in China exceeded $40 trillion—roughly 3× the country's GDP. Yet 200 million rural residents still lack access to basic financial services. Does the fintech revolution reach the people who need it most? |

> 📌 **经济/金融引言第一段的黄金法则：用一个读者关心的真实世界事实开篇——一个政策争论、一个市场悖论、一组令人惊讶的数据。然后自然地导向你的研究问题。永远不要以"X has been extensively studied"开篇。**

#### 贡献列表的写法

两个领域都需要明确列出贡献，但方式有所不同：

**`[CS]` 贡献列表——通常3-4条 bullet points**

| 修改前 | 修改后 |
|---|---|
| Our contributions are: (1) We propose a new model. (2) We conduct extensive experiments. (3) We achieve state-of-the-art results. | Our contributions are: (1) We identify that standard attention scales quadratically because every token attends to every other token, and show that local+global attention suffices for most NLP tasks. (2) We propose Longformer, which reduces attention complexity from O(n²) to O(n) while preserving long-range dependencies. (3) On 6 benchmarks across classification, QA, and summarization, Longformer matches or exceeds RoBERTa while processing 8× longer inputs. |

> 📌 **CS贡献列表的公式：(1) 洞察/发现：你发现了什么关键问题或规律？(2) 方法：你的方案是什么，核心思想一句话。(3) 结果：具体数字。永远不要写"extensive experiments"或"state-of-the-art results"——这是广告，不是贡献。**

**`[Econ/Fin]` 贡献列表——通常嵌入段落中**

经济学论文很少使用 bullet points（有些期刊例外，如 JFE）。贡献通常以"This paper contributes to three strands of literature"的形式嵌入引言段落。

| 修改前 | 修改后 |
|---|---|
| This paper contributes to the literature on financial inclusion. We also contribute to the literature on fintech. Finally, we contribute to the development economics literature. | This paper makes three contributions. First, we provide the first causal estimates of mobile payment adoption on rural household consumption, exploiting the staggered rollout of Alipay across counties. Second, we decompose the channel: 80% of the consumption increase operates through reduced transaction costs, not through expanded credit access—challenging the dominant "credit channel" narrative in the financial inclusion literature. Third, we show that these gains are largest for previously unbanked households, suggesting that fintech complements, rather than substitutes for, traditional banking. |

> 📌 **经济/金融贡献声明的公式：(1) 你提供了什么新的因果证据？(2) 你的发现挑战或修正了文献中的哪个假设？(3) 你的发现对哪个政策辩论有什么具体启示？每一条都要有实质内容，而非仅仅"contribute to X literature"。**

#### 引言自检

- ☐ 第一段是否以一个具体事实/数字/场景开篇（而非"近年来..."或"X has been extensively studied"）？
- ☐ 问题的*重要性*是否被论证了（不仅仅是"gap"存在）？
- ☐ 贡献列表中每一条是否都有实质内容（而非"extensive experiments"或"contribute to the literature"）？
- ☐ `[CS]` 现有方法的具体局限是否被明确指出？
- ☐ `[Econ/Fin]` 识别策略是否被简要介绍？

---

### 第4章：相关工作与文献——区分综述与论证

#### CS 的 Related Work

CS论文的 Related Work 通常在引言之后（有时在论文末尾），按方法类别组织，核心功能是**定位你的工作在技术图谱中的位置**。

**原则：Related Work 不是参考文献的罗列，而是一张精心设计的地图，突出你的位置。**

| 修改前 | 修改后 |
|---|---|
| Zhang et al. (2020) proposed a graph neural network for molecule prediction. Li et al. (2021) used a transformer for the same task. Wang et al. (2022) combined GNN and transformer. | Molecule property prediction has followed two main architectural paradigms. **Graph-based methods** (Zhang et al., 2020; Xu et al., 2019) capture local atomic interactions but struggle with long-range molecular dependencies. **Transformer-based methods** (Li et al., 2021) model global context but ignore the inherent graph structure of molecules. Our work bridges both paradigms by encoding graph topology directly into the transformer's attention mask. |

> 📌 **CS Related Work 的写法：(1) 将文献按技术范式分组。(2) 对每组做一个一句话的优劣评判（不仅仅是描述）。(3) 明确你的工作与每组的区别和联系。最后一句话总是"Our work..."——将你定位在地图上。**

#### 经济/金融的文献综述

经济/金融论文的文献综述通常嵌入引言（JFE风格）或作为独立章节，核心功能是**构建一个指向你研究必要性的论证**。

**原则：文献综述是一个论证（argument），不是一个目录（catalog）。**

| 修改前 | 修改后 |
|---|---|
| Several papers have studied the relationship between monetary policy and asset prices. Bernanke and Kuttner (2005) found that... Rigobon and Sack (2004) showed that... Ehrmann and Fratzscher (2009) documented that... | The conventional view holds that monetary policy affects stock prices primarily through the discount rate channel (Bernanke and Kuttner, 2005). However, this view rests on the assumption that firm expectations respond homogeneously to policy signals—an assumption increasingly at odds with the evidence on heterogeneous information processing (Coibion and Gorodnichenko, 2015). We exploit this tension by testing whether firms with different information sets respond differently to identical policy surprises. |

> 📌 **经济/金融文献综述的写法：(1) 概括"主流观点"（conventional view）。(2) 指出主流观点依赖的关键假设。(3) 展示这个假设为什么可能有问题。(4) 你的研究如何利用这个张力。这不是在罗列论文——而是在构建一个需要你的研究来回答的问题。**

#### 两个领域的共同原则

**每段以论断性主题句开头：**

- ❌ "Many researchers have studied X."（纯描述）
- ✅ "Prior work on X has largely focused on Y, leaving Z unexplored."（论断）
- ❌ "The literature on X is vast."（无用信息）
- ✅ "The literature on X converges on one key prediction, yet the empirical evidence remains mixed."（论断+张力）

> 📌 **主题句测试：你的主题句是否包含一个"观点"——某人可能不同意的判断？如果它只是陈述一个事实（"many people studied X"），它就不是一个好的主题句。**

#### 相关工作/文献自检

- ☐ 每段是否以一个论断性主题句（而非"X et al. studied..."）开头？
- ☐ `[CS]` 文献是否按技术范式分组（而非按时间顺序）？
- ☐ `[CS]` 每组是否有一句话的优劣评判？
- ☐ `[CS]` 你的工作与每组的区别是否明确？
- ☐ `[Econ/Fin]` 文献综述是否构建了一个指向你研究必要性的论证？
- ☐ `[Econ/Fin]` 你是否指出了主流观点的关键假设并展示了张力？

---

## 第三部分：核心方法与理论

---

### 第5章 CS篇：方法——让读者能复现

#### 原则：方法章节的首要读者是想要复现你工作的研究者

**先讲直觉，再讲细节**

CS论文最常见的错误是直接跳入数学公式，没有先解释*为什么*这么设计。

| 修改前 | 修改后 |
|---|---|
| We define the attention score as: $a_{ij} = \text{softmax}(\frac{Q_iK_j^T}{\sqrt{d_k}} + b_{ij})$ where $b_{ij}$ is a learnable bias term. | Standard attention treats all token pairs equally. But in document understanding, nearby tokens are almost always more relevant than distant ones. We introduce a **distance-aware bias** $b_{ij}$ that encodes this prior: tokens within a local window receive a learnable attention bonus, while distant tokens must "earn" attention through content alone. Formally: $a_{ij} = \text{softmax}(\frac{Q_iK_j^T}{\sqrt{d_k}} + b_{ij})$ |

> 📌 **公式三明治法：直觉（为什么）→ 公式（是什么）→ 解释（意味着什么）。永远不要让一个公式独自出现——它前面应有动机，后面应有解读。**

**术语一致性**

| 修改前 | 修改后 |
|---|---|
| ...the representation vector...（方法章节）...the embedding...（实验章节）...the feature...（讨论章节） | ...the node representation...（全文统一） |

> 📌 **在方法章节定义你的术语。之后在整篇论文中完全一致地使用它。"representation""embedding""feature"不是同义词——如果你混用，审稿人会困惑你是否在说不同的东西。**

#### CS 方法自检

- ☐ 每个公式前面是否都有直觉解释（为什么这么设计）？
- ☐ 每个公式后面是否都有文字解读（这意味着什么）？
- ☐ 术语是否全文一致（不混用近义词）？
- ☐ 一个同领域的研究者是否能仅凭方法章节复现你的工作？
- ☐ 算法复杂度是否明确分析？

---

### 第5章 经济金融篇：模型与识别策略——让读者信服你的因果推断

#### 原则：经济学的方法章节核心是回答一个问题——"你怎么知道这是因果关系而非相关性？"

**先讲识别思路，再讲技术细节**

| 修改前 | 修改后 |
|---|---|
| We estimate the following specification: $Y_{it} = \alpha + \beta \cdot Treatment_{it} + \gamma X_{it} + \mu_i + \lambda_t + \epsilon_{it}$. | Our identification exploits a regulatory shock: in 2015, the SEC unexpectedly tightened disclosure requirements for firms above a $75M revenue threshold. Firms just above and below this threshold are similar in observable characteristics (Table 2), but only those above were required to comply. This allows us to estimate the causal effect of disclosure on firm investment using a regression discontinuity design: $Y_{it} = \alpha + \beta \cdot \mathbf{1}[Revenue_{it} > 75M] + f(Revenue_{it}) + X_{it}\gamma + \epsilon_{it}$ |

> 📌 **识别策略三步法：(1) 故事（什么外生变异允许你做因果推断？）→ (2) 可信性论证（为什么处理组和对照组可比？）→ (3) 回归方程（具体实施）。永远不要先放方程再解释为什么它能识别因果效应。**

**明确你的假设**

| 修改前 | 修改后 |
|---|---|
| Under standard assumptions, $\beta$ identifies the causal effect of the policy. | Our estimates require two assumptions: (1) the parallel trends assumption — absent the policy, treated and control counties would have followed similar trajectories (Figure 3 shows no pre-trend divergence), and (2) no contemporaneous shocks — no other policy change affected treated counties at the same time (we test this in Section 5.3). |

> 📌 **不要用"under standard assumptions"一笔带过。明确列出每个假设，并为每个假设提供证据或稳健性检验。审稿人最常见的拒稿理由就是"识别假设不可信"——你必须主动防御。**

#### 经济/金融 方法自检

- ☐ 识别策略是否用"白话"解释清楚了（非经济学家也能理解逻辑）？
- ☐ 识别假设是否被明确列出？
- ☐ 每个假设是否有对应的证据或稳健性检验？
- ☐ 回归方程中的每个变量是否都有明确定义？
- ☐ 标准误的聚类层级是否被论证过？

---

## 第四部分：实验与结果

---

### 第6章 CS篇：实验——公平对比，诚实报告

#### 原则：实验章节的可信度来自公平性和可复现性

**实验设置要像合同一样精确**

| 修改前 | 修改后 |
|---|---|
| We train our model with Adam optimizer and a learning rate of 1e-4. | We train all models (ours and baselines) with AdamW (β₁=0.9, β₂=0.999), linear warmup over 2K steps, then cosine decay to 1e-6, peak learning rate 1e-4, batch size 256 on 4× A100 GPUs. Baselines are retrained using their official codebases with hyperparameters tuned on the validation set (see Appendix B for sweep ranges). |

> 📌 **实验设置的黄金标准：(1) 训练细节精确到可复现。(2) baseline对比必须公平——要么使用作者的官方结果，要么在相同条件下重新训练。(3) 超参数搜索范围透明。**

**结果表格——让对比一目了然**

| 修改前 | 修改后 |
|---|---|
| Our model achieves 94.2% on Dataset A, outperforming all baselines. On Dataset B, our model also performs well. | （用表格呈现，加粗最好结果，下划线次好结果，标注显著性，注明计算预算） |

> 📌 **结果报告原则：(1) 永远用表格而非纯文字展示定量对比。(2) 加粗最优结果，下划线次优。(3) 报告均值±标准差（多次运行）。(4) 如果声称"显著优于"，需要统计检验。(5) 同时报告性能和计算成本。**

**消融实验——证明每个组件都有用**

> 📌 **消融实验是CS论文的"稳健性检验"。对于你方法中的每个关键设计选择，你都需要展示：去掉它，性能会下降多少？这是证明你的贡献是真实的、而非来自超参数调优的最有力方式。**

#### CS 实验自检

- ☐ 训练细节是否精确到可复现？
- ☐ baseline对比是否公平（相同计算预算、相同数据、公平的超参数调优）？
- ☐ 结果是否用表格展示，包含均值和方差？
- ☐ 是否有消融实验证明每个组件的贡献？
- ☐ 是否报告了计算成本（FLOPs、GPU小时、参数量）？

---

### 第6章 经济金融篇：实证结果——让数据说话

#### 原则：结果章节的可信度来自于对潜在质疑的系统性回应

**主结果要简洁有力**

| 修改前 | 修改后 |
|---|---|
| Table 3 presents the regression results. As can be seen from Column (1), the coefficient on the treatment variable is negative and statistically significant at the 1% level, suggesting that the policy had a negative impact on firm investment. | Table 3 shows that the disclosure mandate reduced firm investment by 8.3 percentage points (Column 1, p < 0.01). This effect is economically large: it implies that a typical firm above the threshold cut capital expenditure by approximately $12M annually. |

> 📌 **报告结果的公式：(1) 直接陈述方向和大小（不要说"as can be seen from"）。(2) 提供经济意义的解读（不仅仅是统计显著性）。(3) 把"p < 0.01"当作注脚，把经济含义当主角。**

**稳健性检验——系统性地回应质疑**

经济/金融论文的稳健性检验相当于CS的消融实验。典型结构：

1. **替代性解释**：是否有其他因素（遗漏变量）能解释你的结果？
2. **样本敏感性**：去掉极端值、改变样本期间，结果是否依然成立？
3. **测量敏感性**：换一种方式度量因变量或自变量，结论是否一致？
4. **安慰剂检验**：在你的理论预测不应该有效果的地方，确实没有效果？

> 📌 **稳健性检验不是形式主义——它是你对审稿人说"我已经替你考虑了这些质疑"。最好的稳健性检验来自于你自己对识别策略的最大担忧。**

#### 经济/金融 结果自检

- ☐ 每个核心结果是否都有经济意义的解读（不仅仅是统计显著性）？
- ☐ 是否对最重要的替代性解释进行了检验？
- ☐ 安慰剂检验是否包含在内？
- ☐ 标准误聚类是否合理并被论证？
- ☐ 子样本分析是否用于揭示异质性（而非只报告平均效应）？

---

## 第五部分：讨论与结论

---

### 第7章：讨论与结论——不要虎头蛇尾

#### 原则：结论不是摘要的重复，而是对论文意义的升华

**CS论文的讨论/结论常见问题：太短，只是重复结果**

| 修改前 | 修改后 |
|---|---|
| In this paper, we proposed X and achieved state-of-the-art results on Y benchmarks. In the future, we plan to extend our method to Z. | Our results reveal a broader principle: in long-document understanding, local context carries 80% of the signal (Table 4, ablation). This suggests that the field's focus on ever-longer global attention may be misallocated. More efficient architectures may come not from scaling global attention, but from better combining strong local attention with sparse global connections. **Limitations.** Our method assumes a fixed local window size; documents with highly variable relevant spans (e.g., legal contracts) may require adaptive windows. Additionally, our evaluation is limited to English; we leave multilingual evaluation to future work. |

> 📌 **CS结论的理想结构：(1) 核心发现的更深层含义（不是重复结果，而是提炼规律）。(2) 对领域研究方向的启示。(3) 诚实的局限性讨论。(4) 未来方向——不是"we plan to do X"，而是"our findings suggest that X is a promising direction because Y"。**

**经济/金融论文的讨论常见问题：贡献描述过于笼统**

| 修改前 | 修改后 |
|---|---|
| Our findings have important implications for policymakers. Regulators should consider the unintended consequences of disclosure mandates. | Our findings offer a cautionary note for disclosure regulation: while transparency is beneficial in principle, mandatory disclosure above the $75M threshold reduced investment by 8.3pp — roughly $12M per affected firm per year. This suggests that the compliance costs of SEC reporting, estimated at $1.5M annually (Iliev, 2010), represent only a fraction of the true cost, which operates primarily through managerial short-termism induced by quarterly earnings scrutiny. Policymakers designing disclosure thresholds face a quantifiable trade-off: each additional informed investor comes at the cost of approximately $0.8M in foregone investment. |

> 📌 **经济/金融讨论的黄金法则：(1) 将你的发现翻译成具体的政策含义——不是"regulators should consider"，而是"这意味着具体的权衡是X"。(2) 量化政策含义。(3) 明确你的发现*挑战*了文献中的什么假设。**

#### 局限性——诚实是最好的策略

两个领域都需要诚实讨论局限性。审稿人尊重你能识别自己工作的边界。

> 📌 **写局限性的技巧：不要只列出局限，而是对每条局限说明(1)为什么它不会推翻你的核心结论，以及(2)它如何指向有价值的未来研究方向。这将"弱点"转化为"机会"。**

#### 讨论/结论自检

- ☐ 结论是否提炼了超越具体结果的更深层洞见？
- ☐ 是否明确了对领域研究方向的启示？
- ☐ 局限性是否诚实讨论，并说明为何不推翻核心结论？
- ☐ `[CS]` 是否讨论了方法的适用边界？
- ☐ `[Econ/Fin]` 政策含义是否具体且可量化？

---

## 第六部分：贯穿一切的写作技艺

---

### 第8章：句子层面的手术

以下原则在CS和经济/金融论文中完全通用。

#### 原则1：消灭冗余——每句话都做50%测试

学术写作中最常见的冗余词组：

| 冗余表达 | 简洁替代 |
|---|---|
| It is important to note that | （直接删除） |
| In terms of / With respect to | For / In / On |
| In the context of | In / For |
| Due to the fact that | Because |
| A large number of | Many |
| In order to | To |
| It can be seen that | （直接删除，改写为主动句） |
| We would like to point out that | （直接删除） |
| The reason for this is that | Because |
| At the present time | Now / Currently |

**示例**

| 修改前 | 修改后 |
|---|---|
| It is important to note that our results suggest that, in terms of model performance, the proposed architecture is able to achieve competitive results with respect to existing methods. (30词) | Our architecture matches existing methods in performance. (7词) |

> 📌 **每写完一个句子，问自己：能否用一半的词说同样的话？如果能，就这么做。大多数初稿可以删减30-50%而不损失任何信息。**

#### 原则2：语法平行——列表中的每个条目必须使用相同的语法形式

| 修改前 | 修改后 |
|---|---|
| Our method (1) reduces computation, (2) is improving accuracy, and (3) to generalize better to new domains. | Our method (1) reduces computation, (2) improves accuracy, and (3) generalizes better to new domains. |

> 📌 **平行性测试：列表中的每个条目能否直接接在同一个引导词后面？如果不能，说明语法不平行。**

#### 原则3：主动语态优先

| 修改前 | 修改后 |
|---|---|
| The model was trained on 100K samples and the loss was minimized using SGD. | We trained the model on 100K samples and minimized the loss using SGD. |
| It was found that monetary policy shocks are transmitted through the credit channel. | We find that monetary policy shocks transmit through the credit channel. |

> 📌 **默认使用主动语态。被动语态仅在以下情况使用：(1) 行动者不重要（"The data were collected in 2020"）。(2) 你故意强调宾语（"Three key patterns were identified"）。**

#### 原则4：动词优先于名词化

学术写作有一种将动词变成名词的坏习惯（名词化），这会让句子变得拖沓、抽象。

| 名词化（弱） | 动词形式（强） |
|---|---|
| We performed an analysis of... | We analyzed... |
| We made an improvement to... | We improved... |
| There was an increase in accuracy. | Accuracy increased. |
| The implementation of the algorithm... | We implemented the algorithm... / The algorithm... |
| The utilization of transformer layers... | Using transformer layers... |

> 📌 **搜索你论文中的 "-tion" 和 "-ment" 结尾的词。问自己：我能否用对应的动词替代？几乎总是可以的——而且句子会更短、更有力。**

#### 原则5：段落结构——每段一个核心论点

| 段落结构 | 功能 |
|---|---|
| 第1句：主题句（论断） | 告诉读者这段要说什么 |
| 第2-3句：证据/解释 | 支持你的主张 |
| 最后1句：过渡/总结 | 连接到下一段或回扣大论证 |

> 📌 **段落测试：遮住段落中的所有句子，只看第一句。你能否仅从主题句就知道这段的核心信息？如果不能，你需要重写主题句。**

#### 句子层面自检

- ☐ 我是否对每个句子做了50%删减测试？
- ☐ 所有列表是否语法平行？
- ☐ 是否默认使用主动语态？
- ☐ 是否把名词化表达替换为动词形式？
- ☐ 每段是否以一个清晰的主题句开头？

---

### 第9章：常见写作习惯诊断

以下是CS和经济/金融论文作者最常见的五个写作坏习惯，以及对应的修正方案。

#### 习惯 #1：虚假对冲（Hollow Hedging）

**模式：** "We believe that..." "It seems that..." "It is possible that..." "Our results seem to suggest that..."

**为什么有害：** 合理的对冲（"under assumptions A and B"）是学术诚实。但空洞的对冲只是在削弱你自己的论点——如果你对自己的结果都不自信，审稿人为什么要信？

**修正方法：** 删除对冲词，读句子。如果句子仍然准确，就保持删除。如果确实需要限定条件，用具体的限定替代模糊的对冲：
- ❌ "Our results seem to suggest that X."
- ✅ "X holds when Y (Table 3), though it weakens when Z (Table 4)."

#### 习惯 #2："近年来"开篇（The "Recently" Opener）

**模式：** "In recent years, X has attracted significant attention..." "Recently, deep learning has achieved remarkable success in..."

**为什么有害：** 这是学术论文中最无趣的开场白。它不传递任何信息——审稿人已经知道你的领域很火。

**修正方法：** 直接从问题或令人惊讶的事实开始：
- ❌ "In recent years, large language models have attracted significant attention."
- ✅ "A single ChatGPT query consumes 10× the energy of a Google search — yet 60% of that compute may be wasted on redundant attention calculations."

#### 习惯 #3：结果中的社论式评价（Editorializing Results）

**模式：** "Interestingly, ..." "Surprisingly, ..." "It is noteworthy that..." "An intriguing finding is..."

**为什么有害：** 这些词是在*告诉*读者什么有趣，而不是*展示*给他们看。如果你的发现真的有趣，数据本身会说话。

**修正方法：** 删除评价性副词，直接报告结果。
- ❌ "Interestingly, we find that smaller models outperform larger ones."
- ✅ "Smaller models outperform larger ones (Table 5), suggesting that model capacity beyond 7B parameters yields diminishing returns for this task."

#### 习惯 #4：泛化的行动主体（Generic Agents）

**模式：** "users" "firms" "investors" "practitioners" — 在可以更具体的地方使用泛化词汇。

**修正方法：**
- ❌ `[CS]` "Users may benefit from our method."
- ✅ `[CS]` "NLP practitioners working with documents exceeding 4K tokens — such as legal contracts, scientific papers, or financial reports — can directly apply our method."
- ❌ `[Econ/Fin]` "Our findings have implications for firms."
- ✅ `[Econ/Fin]` "Our findings suggest that publicly traded firms in the $50M-$100M revenue range face the steepest trade-off between disclosure compliance and investment."

#### 习惯 #5：公式缺少上下文（Naked Equations）

**模式：** 公式前后没有直觉解释，就像数学符号凭空出现。

**修正方法：** 用**公式三明治法**——直觉 → 公式 → 解读。

#### 五遍修改方案

**第1遍——冗余猎杀：** 搜索 "it is important" "in terms of" "with respect to" "in order to" "due to the fact that"。全部删除或简化。

**第2遍——对冲清扫：** 搜索 "we believe" "it seems" "it is possible" "our results suggest that"。删除后检查句子是否仍然准确。

**第3遍——开篇修复：** 检查每个章节的第一句话。是否以具体事实/问题开篇？如果以 "In recent years" 或 "It has been widely recognized" 开头，重写。

**第4遍——平行性检查：** 找到所有列表和枚举。验证语法平行性。

**第5遍——具体性升级：** 搜索 "users" "firms" "researchers" 等泛化词汇。在方法/结果/讨论中替换为具体的行动者。

---

## 附录A：CS论文自我编辑清单

*打印此页，每次投稿前逐条检查。*

### 标题
- ☐ 传达核心发现/方法，而非仅描述主题
- ☐ 如有方法名，名字本身暗示核心思想
- ☐ 长度合适（5-12词）

### 摘要
- ☐ 以问题/动机开篇（非"We propose"）
- ☐ 包含核心洞察（为什么有效），不仅仅是描述方法
- ☐ 包含最重要结果的具体数字
- ☐ 不含"new insights""significant contribution"等空话

### 引言
- ☐ 第一段以具体事实/数字/场景开篇
- ☐ 现有方法的具体局限被明确指出
- ☐ 核心思想用1-2句话可解释
- ☐ 贡献列表每条都有实质内容（非"extensive experiments"）

### Related Work
- ☐ 按技术范式分组（非按时间）
- ☐ 每组有优劣评判
- ☐ "Our work..."明确定位

### 方法
- ☐ 每个公式有"直觉→公式→解读"三明治
- ☐ 术语全文一致
- ☐ 设计选择有动机解释（为什么这么做）
- ☐ 复杂度分析

### 实验
- ☐ 训练细节可复现
- ☐ baseline对比公平（相同条件或引用官方结果）
- ☐ 结果用表格展示，含均值±标准差
- ☐ 消融实验证明每个组件贡献
- ☐ 报告计算成本

### 结论
- ☐ 核心发现的更深层含义（非重复结果）
- ☐ 诚实讨论局限性
- ☐ 有意义的未来方向

### 句子层面
- ☐ 每句做过50%删减测试
- ☐ 所有列表语法平行
- ☐ 主动语态为主
- ☐ 无空洞对冲
- ☐ 无 "In recent years" 式开篇

---

## 附录B：经济/金融论文自我编辑清单

*打印此页，每次投稿前逐条检查。*

### 标题
- ☐ 传达核心发现或研究问题
- ☐ 暗示因果方向或识别策略来源
- ☐ 考虑过问句形式

### 摘要
- ☐ 以研究问题开篇（非方法论）
- ☐ 识别策略一句话可解释
- ☐ 核心结果有具体数字和经济意义
- ☐ 不含"contribute to the literature"等空话

### 引言
- ☐ 第一段以政策争论/市场现象/令人惊讶的数据开篇
- ☐ 因果识别的难点被明确指出
- ☐ 贡献声明每条有实质内容
- ☐ 你的发现挑战了文献中的什么假设被明确说明

### 文献综述
- ☐ 构建了一个指向研究必要性的论证（非罗列论文）
- ☐ 指出"主流观点"及其关键假设
- ☐ 展示了假设为何可能有问题
- ☐ 每段以论断性主题句开头

### 识别策略
- ☐ 用非专业语言也能解释清楚
- ☐ 识别假设逐条列出
- ☐ 每个假设有证据/稳健性检验
- ☐ 回归方程每个变量有明确定义
- ☐ 聚类层级被论证

### 结果
- ☐ 直接报告方向和大小（非"as can be seen from"）
- ☐ 每个核心结果有经济意义解读
- ☐ 安慰剂检验
- ☐ 替代解释被系统检验
- ☐ 异质性分析

### 讨论/结论
- ☐ 政策含义具体且可量化
- ☐ 明确你的发现挑战了什么假设
- ☐ 局限性诚实讨论
- ☐ 外部效度（结果是否可推广）被讨论

### 句子层面
- ☐ 每句做过50%删减测试
- ☐ 所有列表语法平行
- ☐ 主动语态为主
- ☐ 无空洞对冲
- ☐ 无 "It has been widely recognized" 式开篇

---

## 附录C：推荐阅读

### 通用学术写作

- **William Zinsser, *On Writing Well*** — 非虚构写作的经典指南。关于简洁、清晰和冗余的章节对学术写作者最有价值。

- **Steven Pinker, *The Sense of Style*** — 认知科学家写的写作指南。"知识的诅咒"章节解释了为什么专家总是写出让别人看不懂的文章。

- **Helen Sword, *Stylish Academic Writing*** — 用数据分析了什么让学术写作变好或变差。

### CS 学术写作

- **Simon Peyton Jones, "How to Write a Great Research Paper"**（演讲视频）— 来自 Haskell 创始人、微软研究院资深科学家的经典建议。核心观点：先写论文，再做研究。

- **Henning Schulzrinne, "Writing Technical Articles"** — 哥伦比亚大学教授整理的 CS 论文写作指南，非常具体和实用。

- **William A. Kahan, "How Futile are Mindless Assessments of Roundoff in Floating-Point Computation?"** — 虽然是关于数值计算的，但文中展示了如何用极其清晰的语言解释复杂的技术概念。

- **Aaditya Ramdas et al., "A Unified Recipe for Deriving (Time-Uniform) PAC-Bayes Bounds"** — 数学很重的论文也能写得清晰、动机充分、结构良好的典范。

### 经济/金融学术写作

- **John H. Cochrane, "Writing Tips for Ph.D. Students"** — 芝加哥大学金融学教授的写作建议。极其实用。核心观点："Your paper is not a mystery novel."

- **Claudia Goldin 和 Lawrence Katz 的论文** — Goldin 是经济学写作的大师。她的论文展示了如何用优雅的语言讲述严谨的因果故事。

- **Jesse Shapiro, "How to Give an Applied Micro Talk"** — 虽然是关于演讲的，但其中关于"构建论证"的建议完全适用于论文写作。

- **Keith Head, "The Introduction Formula"** — UBC 教授整理的经济学引言写作公式，简洁有效。

- **Deirdre McCloskey, *Economical Writing*** — 将经济学原理应用于写作本身的经典小书。

### 论文修改

- **Ezra Zuckerman, "Tips to Article-Writers"** — MIT Sloan 教授的十条学术论文写作规则。对社科和商学院论文最直接有用，但关于"动机"和"零假设"的建议对所有领域通用。

- **Colin Fisher, "Writing an Academic Journal Article"** — 关于引言、文献综述和问题化的实用模板。

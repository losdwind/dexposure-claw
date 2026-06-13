# Overleaf 同步说明（合并了实验部分 + 您的修订）

老师好，我把含完整实验部分的版本合并进了 Overleaf。下面逐节说明**每一节用了谁的版本、为什么**，方便您核对。

**重要兜底**：您同步前的原版完整保存在 Overleaf 的 git 历史 commit `97c134b`，一字未动。任何一处合并决定您若不认同，都能从那里 diff 出来、单独取回。

## 逐节对照

| 章节 | 采用 | 说明 |
|------|------|------|
| Title | **您的版本** | 直接用您改的标题（经匿名宏 `\sysname` 渲染）。 |
| Abstract | **您的版本为骨架** | 保留您的叙事结构；按新的显著性检验把 explanation quality 的措辞降级为 “directionally improves”（见下）。 |
| 1 Introduction | **您的版本为骨架** | 采用您五段式叙事；补回了三处（见下）。 |
| Preliminaries | **您新增的章节** | 保留，但压缩了篇幅（见下）。 |
| 3 Pipeline | **学生版本** | 您的重写与学生版结构一致，但带了过时参数与 LaTeX 损坏，故保留学生版（见下）。 |
| 4–6 / 附录 / 统计 | **学生版本** | 实验、bench、lessons、附录，外加新增的统计显著性附录。 |

## 三处需要您特别留意的改动

**1. Introduction 补回了三块内容（相对您的重写）**
- 失败模式的双框架：raw-snapshot LLM「保守但弱」(FIR 未定义) vs 喂入预测证据后「over-intervention」(FIR 0.448)。这与表 1 的数据方向一致——over-intervention 只在加入 FM 证据后出现，不在 raw-data agent 上。
- 补回了 5 处文献引用（您的重写版 intro 未含引用）。
- 补回了结尾 “Honest scope / track fit” 段（backtest-only 定位、EMNLP Industry 类目）。

**2. Abstract / Results 按显著性检验做了 hedging**
- 新增 bootstrap 置信区间 + permutation 检验（附录 F）后：Sonnet 4.6 的 judge 提升不显著，故 explanation quality 改为 “directionally improves”，只硬声明显著的 F1 与 FIR。
- 表 1 的 FIR：m1/m2/m5 改为 “—”（未定义），因为它们不发 intervention-level ticket（m2 全程 0/431），原来的 0.000 是分母为零的退化值，并非安全成绩；Sonnet/Gemini 的 0.000 是真零（发了干预且全部命中），予以保留。

**3. Preliminaries 压缩了篇幅（189 → 62 行）**
- 为满足正文 6 页上限做的精简。所有定义（exposure graph、forecaster、monitors、CVaR、FIR、gates）均保留，完整推导也在附录 A/B。
- 这是唯一一处我实质删减了您的文字。若您希望保留完整长度，我可以从 `97c134b` 取回您的原版放回（代价是正文会超 6 页，投稿前再压）。

## 为什么 Pipeline 这节保留学生版

您的 pipeline 重写与学生版**结构完全相同**（四层、五个 monitor、scenario engine 一致），差别只有：
- 参数过时：您版 L=26 周、每次约 \$10；学生版是修正后的 L=42、真实成本。
- zip 在格式转换中出现 LaTeX 损坏：部分下标与 `$` 丢失；`top-1 PageRank` 的引用被错配成 HHI 的 `\citep{Rhoades1993HHI}`，且五个 monitor 被压成四个。

采用重写版会把这些过时数字与引用错误带回正文，故保留了学生版（内容无损失）。

## 其它

- 匿名：main.tex 用单一开关 `\anonfalse`(真名) / `\anontrue`(双盲占位名) 控制全文；**当前是真名**，方便您审阅。投稿前会切回 `\anontrue` 并确认回到 6 页。
- 您的备份文件 `1-intro.1.tex`、`3-pipeline.1.tex`、`ABSTRACT.md`、`OUTLINE.md` 原样保留，未删。

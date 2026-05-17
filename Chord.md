第一乐章：Benchmark Contribution
问题：为什么需要这个 benchmark？

保留 Fig. 3 Evaluation Framework，作为结果章节入口。
它告诉读者：我们不是只测 forecasting，而是测 forecast、warning、calibration、stress、decision、robustness 六个轴。

配套表：

Table_TaskI_Summary: 保留，但标题要改得更强，比如 “Task I evidence for deployable risk monitoring”
Table_b1_forecast_AllHorizons: 移 appendix。正文不需要这么大，它会打散主线。
第二乐章：FM Contribution
问题：FM 到底贡献什么？

不要再用单独 RankCorr 图。改成一个新的 Fig. 5，主题不是 “RankCorr across horizons”，而是：

FM turns forecasting into deployable risk signal

三块内容：

Static rank is hard to beat
RankCorr@h4: Persistence 0.570 vs FM 0.558
作用：诚实承认 persistence baseline 强。

FM adds directional signal
TrendCons@h4: FM 0.628 vs Persistence 0.000
作用：说明 FM 不是靠复制当前图，而是提供未来趋势方向。

FM improves deployability
Robust degradation: FM 0.113 vs Persistence 0.148
Calibration badge: PI coverage 0.912 vs target 0.90
作用：把 FM 和监管/agentic 决策连接起来。

这个图不会和表重复，因为表是数字账本，图是论点压缩：FM 的优势不是单点预测，而是 trend + calibration + robustness。

第三乐章：Agentic Framework Contribution
问题：为什么是 FM + LLM + safety gate，而不是单独模型？

这里让 Fig. 4 Layer-wise Contribution 做主图。它应该是结果章节的 climax。

它讲：

m2 -> m6: 加 FM，F1 +32%，Judge +0.45
m5 -> m6: 加 LLM，F1 +27%
m6 -> m7: 加 safety gate，F1 -3%，Judge +0.21
配套表：

Table3_MethodComparison: 保留正文，作为完整多指标审计表。
Table_b5_decision_CrisisPeriod: 我建议移 appendix，或者缩成一句 “layer-wise numbers are visualised in Fig. 4”。否则 Fig. 4 和 Table_b5 功能重复。
Reliability Coda
最后用表收束，不一定画图：

Table_Ablation: 保留正文。它证明 safety gate / data-health gate / scenario engine 不是装饰。
Table_AlertTimeline: 可以保留正文，证明 agentic monitoring 有 lead time。
Table_b4_stress_Detail: 建议 appendix。它是细节审计，不是主叙事。
最终正文图表编排
我建议正文保留：

Figures:

Fig. 1 System overview
Fig. 3 Evaluation framework
新 Fig. 5 FM deployable-signal evidence
Fig. 4 Agentic layer-wise contribution
Tables:

Benchmark schema / methods table，在 Bench section
Table_TaskI_Summary
Table3_MethodComparison
Table_AlertTimeline
Table_Ablation
移到 appendix/supplementary:

Table_b1_forecast_AllHorizons
Table_b4_stress_Detail
Table_b5_decision_CrisisPeriod，如果 Fig. 4 已经承载 layer-wise story
这样故事会更像你说的“协奏曲”：

Benchmark 给舞台和规则
FM 给可部署信号
LLM 给 decision reasoning
Safety gate 给可信约束
Ablation/Timeline 给可靠性背书
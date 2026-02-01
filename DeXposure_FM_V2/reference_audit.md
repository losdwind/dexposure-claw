# DeXposure-FM 引用/交叉引用审计

- 作用域：`DeXposure_FM_V2/DeXposure-FM.tex` 及其 `\input` 递归包含的文件（即实际会被主文档编译的部分）
- Tex 文件数：10
- 使用到的 cite keys：36
- Bib 中条目数：77
- 定义的 labels：40
- 使用到的 refs：20

## 1) 明确错误/需要处理

### 1.1 缺失的引用 key（.tex 里用了，但 .bib 里找不到）
- `xx`
  - `DeXposure_FM_V2/sections/7-conclusion.tex:18`: `The future DeXposure-FM 2.0 will include an innovative hybrid architecture that combines the current auto-regressive forecasting paradigm \fh{\cite{xx}} with diffusion-based generative modeling \fh{\cite{xx}}. Auto-regressive models are efficient and accurate for point and conditional forecasts, but they can struggle to represent globally coherent network trajectories over long horizons, especially under stress. Diffusion models, by contrast, provide a principled way to learn complex conditional distributions by gradually denoising samples, and have recently shown strong performance in structured generation tasks, such as in diffusion language model \fh{\cite{xx}}. %In DeXposure-FM 2.0, we plan to use an autoregressive backbone to produce fast, state-dependent predictions of key sufficient statistics (e.g., protocol states and coarse exposure aggregates), while a conditional diffusion module refines these into full network realizations—jointly sampling link existence and weights in a way that preserves structural constraints such as sparsity, sectoral organization, and budget/flow consistency. This hybrid design would allow the model to output calibrated predictive distributions (not just point forecasts), support scenario-conditional stress testing via controlled guidance, and improve robustness to regime shifts by explicitly modeling the space of plausible future network configurations rather than committing early to a single trajectory.`
  - `DeXposure_FM_V2/sections/3-formulation.tex:86`: `\item \emph{Metadata lookup:} Tokens are directly matched to issuing protocols using metadata from DefiLlama \fh{\cite{xx}} when available.`

### 1.2 未定义的交叉引用（\ref 等指向了不存在的 \label）
- （无）

### 1.3 重复的 \label（同一个 label 在编译图里定义多次）
- （无）

## 2) “不必配的”候选（未被使用）

### 2.1 未被引用的 bib 条目（在 .bib 里，但主文档从未 cite）
- 共 42 条
- `Acemoglu2013` (DeXposure_FM_V2/sections/DeXposure-FM.bib:387)
- `Adamyk2025` (DeXposure_FM_V2/sections/DeXposure-FM.bib:205)
- `Aldasoro2024StablecoinsMMF` (DeXposure_FM_V2/sections/DeXposure-FM.bib:513)
- `Barbon2025DeFiyingFed` (DeXposure_FM_V2/sections/DeXposure-FM.bib:524)
- `BommasaniEtAl2021FoundationModels` (DeXposure_FM_V2/sections/DeXposure-FM.bib:613)
- `BoxJenkins1970` (DeXposure_FM_V2/sections/DeXposure-FM.bib:440)
- `Briola2022TerraLuna` (DeXposure_FM_V2/sections/DeXposure-FM.bib:575)
- `Carriero2025` (DeXposure_FM_V2/sections/DeXposure-FM.bib:9)
- `Daian2019FlashBoys2` (DeXposure_FM_V2/sections/DeXposure-FM.bib:554)
- `Das2023TimesFM` (DeXposure_FM_V2/sections/DeXposure-FM.bib:32)
- `DieboldMariano1995` (DeXposure_FM_V2/sections/DeXposure-FM.bib:409)
- `Ding2024` (DeXposure_FM_V2/sections/DeXposure-FM.bib:1)
- `DuleyEtAl2023Oracle` (DeXposure_FM_V2/sections/DeXposure-FM.bib:249)
- `FSB2023` (DeXposure_FM_V2/sections/DeXposure-FM.bib:187)
- `Faw2025` (DeXposure_FM_V2/sections/DeXposure-FM.bib:25)
- `Foret2021SAM` (DeXposure_FM_V2/sections/DeXposure-FM.bib:218)
- `ForetEtAl2021SAM` (DeXposure_FM_V2/sections/DeXposure-FM.bib:620)
- `GaiKapadia2010Contagion` (DeXposure_FM_V2/sections/DeXposure-FM.bib:354)
- `Glasserman2016` (DeXposure_FM_V2/sections/DeXposure-FM.bib:376)
- `Hollmann2025` (DeXposure_FM_V2/sections/DeXposure-FM.bib:95)
- `HyndmanAthanasopoulos2021` (DeXposure_FM_V2/sections/DeXposure-FM.bib:431)
- `IOSCO2023DeFi` (DeXposure_FM_V2/sections/DeXposure-FM.bib:261)
- `LeharParlourZoican2024FragmentationDEX` (DeXposure_FM_V2/sections/DeXposure-FM.bib:533)
- `Lutkepohl2005` (DeXposure_FM_V2/sections/DeXposure-FM.bib:459)
- `Mazumder2010` (DeXposure_FM_V2/sections/DeXposure-FM.bib:479)
- `Rossi2020TGN` (DeXposure_FM_V2/sections/DeXposure-FM.bib:489)
- `Sankar2020DySAT` (DeXposure_FM_V2/sections/DeXposure-FM.bib:505)
- `SasiBrodeskyNassr2023DeFiLiquidations` (DeXposure_FM_V2/sections/DeXposure-FM.bib:272)
- `Schar2021` (DeXposure_FM_V2/sections/DeXposure-FM.bib:598)
- `Tashman2000` (DeXposure_FM_V2/sections/DeXposure-FM.bib:420)
- `TianZhu2025Liquidations` (DeXposure_FM_V2/sections/DeXposure-FM.bib:284)
- `WatskyAllenDaudEtAl2024StablecoinsPrimarySecondary` (DeXposure_FM_V2/sections/DeXposure-FM.bib:586)
- `WernerEtAl2021SoK` (DeXposure_FM_V2/sections/DeXposure-FM.bib:606)
- `Xu2020TGAT` (DeXposure_FM_V2/sections/DeXposure-FM.bib:497)
- `Zhu2025FinCast` (DeXposure_FM_V2/sections/DeXposure-FM.bib:17)
- `aufiero2025mapping` (DeXposure_FM_V2/sections/DeXposure-FM.bib:169)
- `hu2025realtime` (DeXposure_FM_V2/sections/DeXposure-FM.bib:140)
- `liang2024fmts` (DeXposure_FM_V2/sections/DeXposure-FM.bib:116)
- `naifar2025mapping` (DeXposure_FM_V2/sections/DeXposure-FM.bib:149)
- `nber2025ai` (DeXposure_FM_V2/sections/DeXposure-FM.bib:177)
- `oikonomou2025sam` (DeXposure_FM_V2/sections/DeXposure-FM.bib:132)
- `zbandut2025institutionalizing` (DeXposure_FM_V2/sections/DeXposure-FM.bib:161)

### 2.2 未被引用的 labels（定义了但从未 ref/eqref/cref）
- 共 20 条
- `eq:loss-exist` (DeXposure_FM_V2/sections/4b-training.tex:45)
- `eq:loss-node` (DeXposure_FM_V2/sections/4b-training.tex:61)
- `eq:loss-total` (DeXposure_FM_V2/sections/4b-training.tex:69)
- `eq:loss-weight` (DeXposure_FM_V2/sections/4b-training.tex:53)
- `eq:sis` (DeXposure_FM_V2/sections/6-economics.tex:27)
- `fig:dexposure-fm-architecture` (DeXposure_FM_V2/sections/4a-arch.tex:113)
- `sec:conclusion` (DeXposure_FM_V2/sections/7-conclusion.tex:21)
- `sec:dexposure-fm-architecture` (DeXposure_FM_V2/sections/4a-arch.tex:1)
- `sec:training` (DeXposure_FM_V2/sections/4b-training.tex:2)
- `sec:training:loss` (DeXposure_FM_V2/sections/4b-training.tex:37)
- `subsec:forecast` (DeXposure_FM_V2/sections/5-experiments.tex:61)
- `subsec:limitations` (DeXposure_FM_V2/sections/6-economics.tex:115)
- `subsubsec:forward-looking-risk` (DeXposure_FM_V2/sections/6-economics.tex:62)
- `subsubsec:shock-events` (DeXposure_FM_V2/sections/6-economics.tex:78)
- `subsubsec:systemic-risk` (DeXposure_FM_V2/sections/6-economics.tex:20)
- `subsubsec:task1-impl` (DeXposure_FM_V2/sections/5-experiments.tex:64)
- `subsubsec:task1-results` (DeXposure_FM_V2/sections/5-experiments.tex:68)
- `subsubsec:task2-impl` (DeXposure_FM_V2/sections/5-experiments.tex:120)
- `subsubsec:task2-results` (DeXposure_FM_V2/sections/5-experiments.tex:128)
- `tab:training-hparams` (DeXposure_FM_V2/sections/4b-training.tex:113)

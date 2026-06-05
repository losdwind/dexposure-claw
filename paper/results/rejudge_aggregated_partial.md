## Table 1: Cross-family judge robustness (decision = Claude Opus 4.7, original Pipeline)

| Method | Judge | n | Mean quality | Std | Old Haiku 4.5 baseline |
|---|---|---|---|---|---|
| m6_fm_llm | claude-opus-4.8 | 29 | 2.414 | 0.568 | 2.483 |
| m6_fm_llm | google/gemini-2.5-pro | 19 | 2.421 | 1.465 | 2.474 |

## Table 2: Cross-family decision robustness (judge = Claude Opus 4.8)

| Method | Decision | n | F1 | FIR | Judge quality |
|---|---|---|---|---|---|
| m6_fm_llm | claude-sonnet-4.6 | 3 | 0.0394 | 0.0 | 2 |

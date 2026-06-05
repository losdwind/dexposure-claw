## Table 1: Cross-family judge robustness (decision = Claude Opus 4.7, original Pipeline)

| Method | Judge | n | Mean quality | Std | Old Haiku 4.5 baseline |
|---|---|---|---|---|---|
| m2_snapshot_llm | claude-opus-4.8 | 29 | 2.241 | 0.435 | 2.034 |
| m2_snapshot_llm | google/gemini-2.5-pro | 29 | 2.69 | 1.339 | 2.034 |
| m2_snapshot_llm | openai/gpt-5.5 | 29 | 1.897 | 0.31 | 2.034 |
| m6_fm_llm | claude-opus-4.8 | 29 | 2.414 | 0.568 | 2.483 |
| m6_fm_llm | google/gemini-2.5-pro | 29 | 2.552 | 1.478 | 2.483 |
| m6_fm_llm | openai/gpt-5.5 | 29 | 2.448 | 0.572 | 2.483 |
| m7_fm_llm_gated | claude-opus-4.8 | 29 | 2.448 | 0.572 | 2.69 |
| m7_fm_llm_gated | google/gemini-2.5-pro | 29 | 2.897 | 1.52 | 2.69 |
| m7_fm_llm_gated | openai/gpt-5.5 | 29 | 2.448 | 0.506 | 2.69 |

## Table 2: Cross-family decision robustness (judge = Claude Opus 4.8)

| Method | Decision | n | F1 | FIR | Judge quality |
|---|---|---|---|---|---|
| m6_fm_llm | claude-sonnet-4.6 | 29 | 0.0276 | 0.0 | 2.517 |
| m6_fm_llm | google/gemini-2.5-pro | 29 | 0.0129 | 0.0 | 2.207 |
| m7_fm_llm_gated | claude-sonnet-4.6 | 29 | 0.0288 | 0.0 | 2.655 |
| m7_fm_llm_gated | google/gemini-2.5-pro | 29 | 0.0139 | 0.0 | 2.379 |

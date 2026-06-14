# EMNLP 2026 Industry Track — 6-page Outline

Working title (anonymized for submission):
**"Forecast-Grounded LLM Agents for Financial-Network Supervision:
A Deployed Pipeline, Benchmark, and Operating-Point Audit"**

Target venue: EMNLP 2026 Industry Track (6 pages content + unlimited
references / Limitations / Ethics / Appendix; no ARR; deadline 2026-06-16)

Target reader: an NLP industry reviewer who cares about
(a) where the LLM sits in a deployed pipeline, (b) what the evaluation
actually measures, (c) what the deployment trade-offs are.

---

## Re-framing: how the story changes vs the TKDE version

TKDE positions the paper as "forecast-grounded regulatory framework for
DeFi exposure networks", with the graph foundation model (DeXposure-FM)
as the headline contribution. For EMNLP Industry we invert this:

| Aspect          | TKDE version (current)                                  | EMNLP-Industry version                                                                |
|-----------------|---------------------------------------------------------|---------------------------------------------------------------------------------------|
| Headline        | Graph time-series FM + framework                        | Deployed LLM agent + how to evaluate it                                               |
| FM role         | Main contribution                                       | Upstream "structured evidence provider" for the LLM                                   |
| LLM role        | Decision layer                                          | Central reasoning + ticket-generation + rationale producer                            |
| Benchmark       | Bench for FM+LLM pipelines                              | Bench for LLM-agent evaluation in structured-evidence settings                        |
| Evaluation      | Six axes equally                                        | Lead with b5_decision (ticket F1) + LLM-as-judge + grounding + FIR                    |
| Safety gates    | Engineering detail                                      | First-class deployment lesson (data-health, confidence, audit logs)                   |
| Lesson          | FM helps trend/calibration/robustness                   | LLM agents over structured evidence trade precision for recall and explanation        |

Important: do NOT rename the technical contributions, only the framing. The
underlying maths and experiments stay identical, so we preserve provenance
and avoid Paper-Integrity-Policy risk.

---

## 6-Page Layout (ACL single-column template)

Rough budgeting assumes ACL 2026 template at default font/spacing
(~50 lines/column, single column).

### Page 1 — Abstract + Section 1 (Introduction)

- Title, anonymous authors, abstract (~180 words; see ABSTRACT.md)
- 1 Introduction (~3/4 page)
  - Hook: DeFi crisis story (Terra/Luna, FTX) compressed to one paragraph
  - Problem framing reframed as *NLP/agent problem*:
    "general-purpose LLM agents over raw on-chain data produce
    hallucinated risk narratives; we need a pipeline where the LLM
    reasons over **structured forecasted evidence**, with
    safety gates and an auditable rationale trail."
  - Three bullet contributions, but ordered for NLP audience:
    1. **Deployed LLM-agent pipeline** with safety gating + audit logs
       (the system; what we run weekly on real DeFi data)
    2. **An evaluation harness** for LLM agents that reason over
       structured time-series evidence, with 8 reference methods
       and regulator-aligned absolute-loss ground truth
    3. **Operating-point audit**: F1 +208%, judge score +33%,
       at a measured false-intervention cost — characterised as a
       Pareto option, not a uniform win

### Page 2 — Section 1 cont'd + Section 2 (Related Work, compressed)

- Finish contributions + roadmap (~1/4 page)
- 2 Related Work (~3/4 page, *very* compressed — Industry Track tolerates
  much less related work than Main):
  - LLM agents for finance and ops (cite Arbiter, KnowYourIntent, etc.)
  - LLM evaluation benchmarks (HELM, AgentBench, SWE-bench) → gap on
    domain-grounded supervisory decision quality
  - Graph foundation models, *briefly*, with citation to TKDE-version
    technical details (anonymous self-citation)
- ~1 short paragraph each, no subsections

### Page 3 — Section 3 (Deployed Pipeline)

- 3 Deployed Pipeline (full page; includes figure)
  - Figure 1: 4-layer pipeline diagram (reuse fig1_system_overview.tex)
  - 3.1 Interface to the FM (1 paragraph; treat as black-box forecast
    primitive, defer maths to Appendix A)
  - 3.2 Structured evidence: monitor signals (N1-N5) + scenario engine,
    summarised as "5 metrics, 5 stress scenarios, evidence bundle with
    attribution top-K". Defer full equations to Appendix B.
  - 3.3 LLM decision layer:
    - Input format (structured JSON evidence bundle)
    - Output format (ticket with severity, target, rationale)
    - Grounding constraint and self-consistency probe
  - 3.4 Safety gates: data-health + confidence; safe-mode behaviour
    (1 paragraph; bullet the 4 actions: Monitor/Investigate/
    Recommend-Reduce/Contingency)

### Page 4 — Section 4 (Benchmark) + start of Section 5 (Results)

- 4 Benchmark (~half page)
  - Why a new bench: gap from HELM/AgentBench/TGB (1 paragraph)
  - Schema table (compressed to 3 columns: ID / what it tests /
    primary metric)
  - Regulator-aligned absolute-loss ground truth (1 paragraph,
    the equation goes to Appendix C)
  - 8 reference methods — *short* bullet list, full method table
    in Appendix D
- 5 Empirical Results (~half page; the central table is here)
  - Headline result table (replace TKDE's Tables 1+2 with a single
    consolidated table: method × {F1, judge, grounding, FIR,
    safe-mode rate, cost-adjusted score})

### Page 5 — Section 5 cont'd

- 5 Results cont'd (full page)
  - Figure 2: layer-wise contribution + cost (reuse fig4_method_comparison)
  - 3 short paragraphs:
    - "Persistence+rules is the precision baseline; FM+LLM triples recall"
    - "FM and LLM make additive, non-substitutable F1 contributions"
    - "Safety gate trades 3% F1 for +0.21 judge score"
  - Ablation table (compressed Panel A only; B/C to Appendix E)
  - 2 short paragraphs:
    - "A2 confidence gate is load-bearing; A3 scenario engine is
      load-bearing for target extraction"
    - "A1 data-health gate is dormant on clean data but reserves
      activate under degraded data (full per-regime audit in App. E)"

### Page 6 — Section 6 (Deployment Lessons) + Section 7 (Conclusion)

- 6 Deployment Lessons (~half page; this is the section a Main-track
  reviewer would skip but Industry-track reviewer wants most)
  - L1: Grounding score 1.000 ≠ correctness — LLM over-reaches
    on severity even with perfect evidence citation
  - L2: Audit logs *as a design artefact*, not just an output —
    every emitted ticket carries the prediction, monitor scores,
    scenario evidence, and rationale that produced it; we release
    them so operators can recompute their own operating point
  - L3: Safety gates as reserves, not always-on filters —
    measurable cost in clean conditions, measurable benefit under
    degradation
  - L4: Cost / cadence: full pipeline fits a weekly supervisory
    cadence on a single RTX 4090 + ~$10/run of LLM API; this is
    what makes it deployable in an academic-budget supervisory
    setting
- 7 Conclusion (~1/4 page; short; reframe as
  "what to take away if you build a similar LLM-agent system")

After page 6 (does not count against page budget):
- Limitations (3 paragraphs from existing 7-Limitations.tex,
  lightly adapted)
- Ethics statement (new, ~half page; see ETHICS.md placeholder below)
- References
- Appendices A-E (everything we cut from main text; see mapping below)

---

## Main-Text → Appendix Mapping

What MUST move to appendix to fit 6 pages (with anchor links from main text):

| Source section (current TKDE)                  | Goes to                  | Appendix label          |
|------------------------------------------------|--------------------------|-------------------------|
| 3-Pre.tex (preliminaries / notation)           | Appendix A               | "FM and notation"       |
| 4-Alg.tex equations + Algorithm 1 pseudocode   | Appendix B               | "Pipeline mathematics"  |
| 5-Bench.tex ground-truth equation, full schema | Appendix C               | "Benchmark details"     |
| 5-Bench.tex reference method table             | Appendix D               | "Reference methods"     |
| 6-Exp.tex per-horizon b1 details (T1, A1)      | Appendix E.1             | "Full b1 results"       |
| 6-Exp.tex per-scenario b4 (A2)                 | Appendix E.2             | "Per-scenario b4"       |
| 6-Exp.tex per-regime b6 (A3)                   | Appendix E.3             | "Per-regime b6"         |
| 6-Exp.tex per-budget b2 (A4)                   | Appendix E.4             | "Per-budget b2"         |
| 6-Exp.tex per-regime A1 isolation (A5)         | Appendix E.5             | "A1 isolation"          |
| Ablation Panels B + C                          | Appendix E.6             | "Stress-condition ablations" |

What STAYS in main text:
- 1 system figure (fig1)
- 1 layer-wise contribution figure (fig4)
- 1 schema table (compressed)
- 1 headline results table (consolidated)
- 1 ablation table (Panel A only)

---

## Anonymisation Checklist

Before submission, scan for and replace:

- [ ] Author block in main .tex (currently lines 38-43 of DeXposure-Agent.tex)
- [ ] Acknowledgments section (currently absent — keep absent until camera-ready)
- [ ] System names "DeXposure-Claw / DeXposure-FM / DeXposure-Bench" — if
      these are public on GitHub/arXiv, replace with placeholders
      ("Claw-System", "Forecast-FM", "Decision-Bench") for the submission
      version; restore in camera-ready
- [ ] All self-citations rewritten to anonymous third-person
      ("Smith (1991) previously showed" → "Previous work showed")
- [ ] Any URLs / repo links / footnotes pointing to the public repo
- [ ] Figure captions that mention "Edinburgh" or any institution
- [ ] meta-data of figure PDFs (run `exiftool -all= *.pdf`)
- [ ] Code-availability statement → use anonymous repo (4open.science)

---

## Files to create (next step, not in this pass)

- main.tex (ACL template skeleton, anonymized title block)
- sections/1-intro.tex (rewritten for NLP-industry framing)
- sections/2-related.tex (compressed)
- sections/3-pre.tex (Preliminaries / notation)
- sections/4-pipeline.tex (pipeline)
- sections/5-bench.tex (benchmark harness)
- sections/6-results.tex (empirical results)
- sections/7-lessons.tex (Deployment Lessons section)
- sections/8-conclusion.tex (short)
- sections/9-limitations.tex (lightly adapted from old Limitations)
- sections/10-ethics.tex
- sections/appendix-*.tex (mapping above)
- references.bib (subset of existing reference.bib that survives in
  the compressed related-work section)

---

## Open questions for next pass

1. Should the system / bench keep its real name in submission? Risk:
   if "DeXposure" is already visible on a public repo or arXiv preprint
   under your name, leaving the name in submission breaks anonymity.
   Need to confirm what is currently public.
2. Do we want Claude as the LLM in the headline results, or do we want
   to add a second LLM (e.g. GPT-4o) for breadth? Industry reviewers
   often ask "does it work with a different model?"
3. Code/data release statement: 4open.science vs anonymized GitHub —
   we should pick one before camera-ready.

# Abstract Drafts for EMNLP 2026 Industry Track

Target: 150-200 words. Anonymous. Lead with the deployed LLM-agent
system and its evaluation, not with the graph foundation model.

**STATUS: SUPERSEDED (2026-06-15).** The authoritative abstract is the one
embedded in `main.tex`, revised to (a) disclose that the Opus 4.7 decision
model still over-reads the forecaster (false-intervention rate ~0.44; 0.437
for the gated stack) so that the safe operating point is the Sonnet 4.6 swap,
and (b) name DeXposure-Bench (six-axis harness, absolute-loss ground truth,
false-intervention rate, eight reference implementations, audit logs) as a
contribution. The drafts below are kept for historical reference only and
their numbers are STALE — in particular the judge claim "2.03 -> 2.69 (+33%)"
and "+208%" predate the honest Section 6 framing, where the m2 -> m7 judge
lift is reported as directional (n.s. under the Opus judge). Do NOT copy
numbers from this file; cite `main.tex` and the Results section instead.

---

## Draft A (FINAL — leads with deployment + evaluation)

General-purpose LLM agents asked to recommend financial-risk
interventions from raw on-chain data tend to hallucinate severity and
over-trigger high-stakes actions. We describe a deployed LLM-agent
pipeline for decentralised-finance supervision in which the LLM
reasons not over raw transactions but over **structured forecasted
evidence**: a graph time-series forecaster produces predicted exposure
networks, deterministic monitors and stress scenarios convert them
into a typed evidence bundle, and the LLM emits supervisory tickets
with rationales. Two safety gates — a data-health gate and a
confidence gate — determine when intervention-level tickets are
allowed; every emitted ticket carries the evidence bundle and rationale
that produced it.
We release an evaluation harness that scores such pipelines on six
axes, including ticket F1 against a regulator-aligned absolute-loss
ground truth, LLM-as-judge explanation quality, and a
false-intervention rate. With eight reference implementations on five
years of weekly DeFi exposure graphs, the full LLM-agent stack raises
ticket F1 from 0.0076 to 0.0233 (+208%) and judge score from 2.03 to
2.69/5 (+33%), while making the resulting false-intervention cost
(FIR≈0.44 vs 0 for rules) visible. We characterise the system as a
recall-explanation Pareto option for supervisory deployment, not a
uniform replacement for conservative rules, and release ticket-level
audit logs.

---

## Draft B (alternative — leads with the evaluation gap)

Benchmarks for LLM agents (HELM, AgentBench, SWE-bench) measure
open-ended reasoning or coding, and temporal-graph benchmarks (TGB,
OGB) stop at structural prediction; neither measures whether an LLM
agent that consumes structured time-series evidence produces
supervisory decisions a regulator would actually accept. We close that
gap with an evaluation harness — six axes, a regulator-aligned
absolute-loss ground truth, and eight reference method implementations
— and apply it to a deployed LLM-agent pipeline for decentralised-finance
risk supervision. In the pipeline, a graph time-series foundation
model produces predicted exposure networks; deterministic monitors and
stress scenarios convert them into a typed evidence bundle; and an LLM
emits supervisory tickets with rationales, gated by a data-health and
a confidence gate. On the harness, the full LLM-agent stack raises
ticket F1 from 0.0076 to 0.0233 (+208%) and LLM-as-judge explanation
quality from 2.03 to 2.69/5 (+33%) over a persistence-plus-rules
baseline, at a measurable false-intervention cost (FIR≈0.44 vs 0). We
release the harness, the eight reference implementations, and the
ticket-level audit logs, and characterise the system as a Pareto
option rather than a uniform replacement.

---

## Why Draft A is preferred for Industry Track

1. **Opens with a real deployment problem** ("LLM agents hallucinate
   severity"). Industry reviewers read the first sentence as the
   problem statement; an evaluation-gap opening reads as a Main-track
   resource paper.
2. **System-first contribution order** (pipeline → harness →
   characterisation) matches the Industry CFP's listed acceptable
   contribution types ("System design and deployment case studies",
   "Methods for deployed systems (evaluation, ethics, robustness)",
   "Human-in-the-loop system design").
3. **Audit logs explicitly named** — reviewers looking for
   "reproducibility / honest deployment" can latch onto this in 5
   seconds.
4. **Pareto framing in the abstract** — pre-empts the obvious
   reviewer concern ("FIR 0.44 vs 0 means your system is worse!").
5. **Names two safety gates by purpose, not by symbol** — keeps the
   abstract readable for an NLP reviewer who has no idea what
   $\tau_{\mathrm{data}}$ or $DH_t$ mean.

---

## What I deliberately removed vs the TKDE abstract

- "Foundation model" is no longer in the first sentence (still mentioned,
  but as a supplier of evidence, not as the headline)
- "DeXposure-FM / DeXposure-Bench / DeXposure-Claw" trade names removed
  for anonymity; restore in camera-ready
- "$43.7$ million data entries" removed from abstract (too domain-y for
  NLP audience; moved to Results)
- "Forecast-grounded regulatory decision framework" replaced by
  "deployed LLM-agent pipeline" — same system, NLP-shaped framing
- Theme-track word "supervision" kept (it's domain-correct), but
  "regulatory" softened to "supervisory" to avoid signalling
  finance-only audience

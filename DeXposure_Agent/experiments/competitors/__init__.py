"""Competitor model wrappers for DeXposure-Agent benchmarks.

Implemented methods (see experiments/methods.py for the canonical registry):

  m1_persistence_rules  Persistence + rule engine (no learned predictor).
  m2_snapshot_llm       Pure LLM baseline reading the current snapshot only.
  m3_evolvegcn          EvolveGCN-O/H (Pareja et al., KDD 2020).
  m4_fm_only            FM predictor without the agent decision layer.
  m5_fm_rules           DeXposure-Agent (FM predictor + rule engine).
  m6_fm_llm             FM predictor + LLM decision agent.
  m7_fm_llm_gated       FM predictor + LLM with rule-based safety gating.

Heuristics:
  h1_weighted_degree    Shared weighted-degree early-warning ranker.
"""

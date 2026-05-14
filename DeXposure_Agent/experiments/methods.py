"""Canonical method registry for DeXposure-Agent experiments.

IDs are sortable slugs. Methods sort alphanumerically in the order they appear
in the paper: persistence baselines < FM-only < FM+rules < FM+LLM variants.
Heuristics use the ``h`` namespace; learned methods use ``m``.

There is no legacy alias here on purpose: this file is the only place ID
strings are defined, and every other module imports from it.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class MethodSpec:
    method_id: str
    label: str
    predictor: str
    policy: str
    implemented: bool
    gpu_required: bool = False
    notes: str = ""


METHODS: dict[str, MethodSpec] = {
    "m1_persistence_rules": MethodSpec(
        method_id="m1_persistence_rules",
        label="Persistence+Rules",
        predictor="persistence",
        policy="rules",
        implemented=True,
    ),
    "m2_snapshot_llm": MethodSpec(
        method_id="m2_snapshot_llm",
        label="CurrentSnapshot+LLM",
        predictor="current",
        policy="llm",
        implemented=True,
    ),
    "m3_evolvegcn": MethodSpec(
        method_id="m3_evolvegcn",
        label="EvolveGCN",
        predictor="evolvegcn",
        policy="none",
        implemented=True,
        gpu_required=True,
        notes="Requires trained EvolveGCN checkpoints; fail closed if absent.",
    ),
    "m4_fm_only": MethodSpec(
        method_id="m4_fm_only",
        label="FMOnly",
        predictor="fm",
        policy="none",
        implemented=True,
        gpu_required=True,
    ),
    "m5_fm_rules": MethodSpec(
        method_id="m5_fm_rules",
        label="FM+Rules",
        predictor="fm",
        policy="rules",
        implemented=True,
        gpu_required=True,
    ),
    "m6_fm_llm": MethodSpec(
        method_id="m6_fm_llm",
        label="FM+LLM",
        predictor="fm",
        policy="llm",
        implemented=True,
        gpu_required=True,
    ),
    "m7_fm_llm_gated": MethodSpec(
        method_id="m7_fm_llm_gated",
        label="FM+LLM+RulesGate",
        predictor="fm",
        policy="llm_rules_gate",
        implemented=True,
        gpu_required=True,
    ),
    "h1_weighted_degree": MethodSpec(
        method_id="h1_weighted_degree",
        label="WeightedDegreeHeuristic",
        predictor="current",
        policy="heuristic",
        implemented=True,
        notes="Method-agnostic b2_warning early-warning heuristic.",
    ),
}


METHOD_NAMES: dict[str, str] = {
    method_id: spec.label for method_id, spec in METHODS.items()
}


def get_method(method_id: str) -> MethodSpec:
    try:
        return METHODS[method_id]
    except KeyError as exc:
        raise ValueError(f"Unknown method_id: {method_id!r}") from exc


def require_implemented(method_id: str) -> MethodSpec:
    spec = get_method(method_id)
    if not spec.implemented:
        raise NotImplementedError(f"{method_id} ({spec.label}) is not implemented")
    return spec

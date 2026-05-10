"""Canonical method registry for DeXposure-Agent experiments.

The paper may still show compact legacy IDs, but experiment code should route
through explicit method compositions so unavailable models cannot silently
masquerade as persistence baselines.
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
    "C0": MethodSpec(
        method_id="C0",
        label="FM+Rules",
        predictor="fm",
        policy="rules",
        implemented=True,
        gpu_required=True,
    ),
    "C2": MethodSpec(
        method_id="C2",
        label="Persistence+Rules",
        predictor="persistence",
        policy="rules",
        implemented=True,
    ),
    "C3": MethodSpec(
        method_id="C3",
        label="CurrentSnapshot+LLM",
        predictor="current",
        policy="llm",
        implemented=True,
    ),
    "C0-LLM": MethodSpec(
        method_id="C0-LLM",
        label="FM+LLM",
        predictor="fm",
        policy="llm",
        implemented=True,
        gpu_required=True,
    ),
    "C0-LLM-GATED": MethodSpec(
        method_id="C0-LLM-GATED",
        label="FM+LLM+RulesGate",
        predictor="fm",
        policy="llm_rules_gate",
        implemented=True,
        gpu_required=True,
    ),
    "C4": MethodSpec(
        method_id="C4",
        label="FMOnly",
        predictor="fm",
        policy="none",
        implemented=True,
        gpu_required=True,
    ),
    "C7": MethodSpec(
        method_id="C7",
        label="EvolveGCN",
        predictor="evolvegcn",
        policy="none",
        implemented=True,
        gpu_required=True,
        notes="Requires trained EvolveGCN checkpoints; fail closed if absent.",
    ),
    "H0": MethodSpec(
        method_id="H0",
        label="WeightedDegreeHeuristic",
        predictor="current",
        policy="heuristic",
        implemented=True,
        notes="Method-agnostic B2 early-warning heuristic.",
    ),
}


LEGACY_UNIMPLEMENTED_METHODS: dict[str, str] = {
    "C1": "ROLAND-Agent",
    "C5": "ROLAND",
    "C6": "GraphPFN-Frozen",
    "C8": "DyRep",
    "C9": "TGN",
    "C10": "Static GCN",
}


METHOD_NAMES: dict[str, str] = {
    **{method_id: spec.label for method_id, spec in METHODS.items()},
    **LEGACY_UNIMPLEMENTED_METHODS,
}


def get_method(method_id: str) -> MethodSpec:
    try:
        return METHODS[method_id]
    except KeyError as exc:
        if method_id in LEGACY_UNIMPLEMENTED_METHODS:
            raise NotImplementedError(
                f"{method_id} ({LEGACY_UNIMPLEMENTED_METHODS[method_id]}) is listed "
                "as a paper baseline but is not implemented in this experiment stack."
            ) from exc
        raise ValueError(f"Unknown method_id: {method_id}") from exc


def require_implemented(method_id: str) -> MethodSpec:
    spec = get_method(method_id)
    if not spec.implemented:
        raise NotImplementedError(f"{method_id} ({spec.label}) is not implemented")
    return spec

#!/usr/bin/env python3
"""Bootstrap CIs + paired permutation tests for the EMNLP paper tables.

Reads per-week artifacts already on disk (no pipeline re-run, no API calls):

  results/run_20260515_pipeline_full/llm_eval/raw_m{2,6,7}_*.json
      -> per-week precision / completeness / FIR for the Opus-4.7 decisions
  results/rejudge_20260603_023028/rejudge_*__<judge>.json
      -> per-week judge scores for the judge panel (Table 2)
  results/redecide_20260603_023030/decide_*__<model>.json
      -> per-week precision / completeness / FIR for Sonnet/Gemini decisions
  results/redecide_20260603_023030/rejudge_*__<model>__claude-opus-4.8.json
      -> per-week Opus-4.8 judge scores for Sonnet/Gemini decisions
  results/local_rerun_m1/b5_weekly__m1_persistence_rules.json (optional)
      -> per-week recall/precision-hits for the m1 rules baseline

Outputs results/bootstrap_stats.json + a human-readable summary to stdout.

F1 convention matches the paper: F1 = 2PR/(P+R) computed on the
across-week MEAN precision and MEAN recall (micro-style aggregate),
bootstrapped by resampling weeks with replacement.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

RESULTS = Path(__file__).resolve().parent.parent / "results"
RNG = np.random.default_rng(20260610)
N_BOOT = 10_000
N_PERM = 20_000


# ---------------------------------------------------------------------------
# Loaders -> per-week arrays
# ---------------------------------------------------------------------------

def load_llm_eval(method: str) -> dict[str, np.ndarray]:
    """Per-week precision/completeness/FIR from the Opus-4.7 llm_eval raws."""
    path = RESULTS / "run_20260515_pipeline_full" / "llm_eval" / f"raw_{method}.json"
    raw = json.load(open(path))
    prec, comp, fir, dates = [], [], [], []
    for e in raw:
        a = e["assessment"]
        prec.append(a["precision"])
        comp.append(a["completeness"])
        fir.append(a["false_intervention_rate"])
        dates.append(e["date"])
    return {"date": np.array(dates), "precision": np.array(prec, float),
            "recall": np.array(comp, float), "fir": np.array(fir, float)}


def load_judge(path: Path) -> dict[str, np.ndarray]:
    d = json.load(open(path))
    ent = sorted(d["entries"], key=lambda e: e["date"])
    return {"date": np.array([e["date"] for e in ent]),
            "score": np.array([e["quality_score"] for e in ent], float)}


def load_decide(method: str, model: str) -> dict[str, np.ndarray]:
    path = RESULTS / "redecide_20260603_023030" / f"decide_{method}__{model}.json"
    d = json.load(open(path))
    ent = sorted(d["entries"], key=lambda e: e["date"])
    prec = np.array([e["assessment"]["precision"] for e in ent], float)
    comp = np.array([e["assessment"]["completeness"] for e in ent], float)
    fir = np.array([e["assessment"]["false_intervention_rate"] for e in ent], float)
    return {"date": np.array([e["date"] for e in ent]),
            "precision": prec, "recall": comp, "fir": fir}


def load_m1_weekly() -> dict[str, np.ndarray] | None:
    path = RESULTS / "local_rerun_m1" / "b5_weekly__m1_persistence_rules.json"
    if not path.exists():
        return None
    d = json.load(open(path))
    weeks = d["weeks"]
    # m1's per-week precision is over its (few) ticket targets; weeks with no
    # tickets contribute no precision observations (micro-averaged upstream).
    rec = np.array([w["recall_at_k"] if w["recall_at_k"] is not None else np.nan
                    for w in weeks], float)
    hits = [w["precision_hits"] for w in weeks]
    fir = np.array([float(np.mean(w["false_interventions"])) if w["false_interventions"]
                    else 0.0 for w in weeks], float)
    return {"date": np.array([w["date"] for w in weeks]),
            "recall": rec, "precision_hits": hits, "fir": fir}


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------

def f1_from_pr(p: float, r: float) -> float:
    return 2 * p * r / (p + r) if (p + r) > 0 else 0.0


def boot_ci_mean(x: np.ndarray, n_boot: int = N_BOOT) -> tuple[float, float, float]:
    """Mean with 95% percentile bootstrap CI, resampling weeks."""
    x = x[~np.isnan(x)]
    idx = RNG.integers(0, len(x), size=(n_boot, len(x)))
    means = x[idx].mean(axis=1)
    return float(x.mean()), float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def boot_ci_f1(prec: np.ndarray, rec: np.ndarray, n_boot: int = N_BOOT) -> tuple[float, float, float]:
    """F1 of mean precision and mean recall, week-level bootstrap."""
    n = len(prec)
    point = f1_from_pr(float(np.nanmean(prec)), float(np.nanmean(rec)))
    idx = RNG.integers(0, n, size=(n_boot, n))
    f1s = np.array([f1_from_pr(float(np.nanmean(prec[i])), float(np.nanmean(rec[i])))
                    for i in idx])
    return point, float(np.percentile(f1s, 2.5)), float(np.percentile(f1s, 97.5))


def paired_perm_test(a: np.ndarray, b: np.ndarray, n_perm: int = N_PERM) -> float:
    """Two-sided paired sign-flip permutation test on mean(a - b)."""
    mask = ~(np.isnan(a) | np.isnan(b))
    d = (a - b)[mask]
    obs = abs(d.mean())
    signs = RNG.choice([-1.0, 1.0], size=(n_perm, len(d)))
    null = np.abs((signs * d).mean(axis=1))
    return float((np.sum(null >= obs) + 1) / (n_perm + 1))


def paired_perm_f1(pa, ra, pb, rb, n_perm: int = N_PERM) -> float:
    """Paired permutation test on the F1-of-means statistic: per week, swap
    method A's and B's (precision, recall) pair or not, recompute both F1s."""
    n = len(pa)
    obs = abs(f1_from_pr(np.nanmean(pa), np.nanmean(ra))
              - f1_from_pr(np.nanmean(pb), np.nanmean(rb)))
    count = 0
    for _ in range(n_perm):
        swap = RNG.random(n) < 0.5
        qa_p = np.where(swap, pb, pa); qa_r = np.where(swap, rb, ra)
        qb_p = np.where(swap, pa, pb); qb_r = np.where(swap, ra, rb)
        stat = abs(f1_from_pr(np.nanmean(qa_p), np.nanmean(qa_r))
                   - f1_from_pr(np.nanmean(qb_p), np.nanmean(qb_r)))
        if stat >= obs:
            count += 1
    return (count + 1) / (n_perm + 1)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    out: dict = {"n_boot": N_BOOT, "n_perm": N_PERM, "ci": {}, "tests": {}}

    # ---- Decision metrics, Opus 4.7 main config (Table 1 LLM rows) ----
    m = {meth: load_llm_eval(meth) for meth in
         ("m2_snapshot_llm", "m6_fm_llm", "m7_fm_llm_gated")}
    for meth, d in m.items():
        f1, lo, hi = boot_ci_f1(d["precision"], d["recall"])
        fm, flo, fhi = boot_ci_mean(d["fir"])
        out["ci"][f"{meth}/f1"] = [f1, lo, hi]
        out["ci"][f"{meth}/fir"] = [fm, flo, fhi]
        print(f"{meth:24s} F1 {f1:.4f} [{lo:.4f},{hi:.4f}]   FIR {fm:.3f} [{flo:.3f},{fhi:.3f}]")

    # ---- Sonnet / Gemini decision panels (Table 3) ----
    panels = {}
    for meth in ("m6_fm_llm", "m7_fm_llm_gated"):
        for model in ("claude-sonnet-4.6", "google_gemini-2.5-pro"):
            d = load_decide(meth, model)
            panels[(meth, model)] = d
            f1, lo, hi = boot_ci_f1(d["precision"], d["recall"])
            out["ci"][f"{meth}+{model}/f1"] = [f1, lo, hi]
            print(f"{meth}+{model:22s} F1 {f1:.4f} [{lo:.4f},{hi:.4f}]")

    # ---- Judge panel scores (Table 2 + decision-panel judges) ----
    jdir = RESULTS / "rejudge_20260603_023028"
    rdir = RESULTS / "redecide_20260603_023030"
    judge_files = {
        ("m2", "opus4.8"): jdir / "rejudge_m2_snapshot_llm__claude-opus-4.8.json",
        ("m6", "opus4.8"): jdir / "rejudge_m6_fm_llm__claude-opus-4.8.json",
        ("m7", "opus4.8"): jdir / "rejudge_m7_fm_llm_gated__claude-opus-4.8.json",
        ("m2", "gemini"): jdir / "rejudge_m2_snapshot_llm__google_gemini-2.5-pro.json",
        ("m6", "gemini"): jdir / "rejudge_m6_fm_llm__google_gemini-2.5-pro.json",
        ("m7", "gemini"): jdir / "rejudge_m7_fm_llm_gated__google_gemini-2.5-pro.json",
        ("m2", "gpt5.5"): jdir / "rejudge_m2_snapshot_llm__openai_gpt-5.5.json",
        ("m6", "gpt5.5"): jdir / "rejudge_m6_fm_llm__openai_gpt-5.5.json",
        ("m7", "gpt5.5"): jdir / "rejudge_m7_fm_llm_gated__openai_gpt-5.5.json",
        ("m6+sonnet", "opus4.8"):
            rdir / "rejudge_m6_fm_llm__claude-sonnet-4.6__claude-opus-4.8.json",
        ("m7+sonnet", "opus4.8"):
            rdir / "rejudge_m7_fm_llm_gated__claude-sonnet-4.6__claude-opus-4.8.json",
    }
    judges = {}
    for key, path in judge_files.items():
        if not path.exists():
            print(f"  !! missing {path}", file=sys.stderr)
            continue
        judges[key] = load_judge(path)
        mean, lo, hi = boot_ci_mean(judges[key]["score"])
        out["ci"][f"judge/{key[0]}/{key[1]}"] = [mean, lo, hi]
        print(f"judge {key[0]:11s} x {key[1]:8s} {mean:.2f} [{lo:.2f},{hi:.2f}]")

    # ---- Paired tests (aligned by date; all share the same 29 weeks) ----
    def aligned(d1, d2, k1, k2):
        common = sorted(set(d1["date"]) & set(d2["date"]))
        i1 = [list(d1["date"]).index(c) for c in common]
        i2 = [list(d2["date"]).index(c) for c in common]
        return d1[k1][i1], d2[k2][i2]

    tests = {
        "f1: m6 vs m2 (FM signal)": ("f1", m["m6_fm_llm"], m["m2_snapshot_llm"]),
        "f1: m7 vs m6 (gate)": ("f1", m["m7_fm_llm_gated"], m["m6_fm_llm"]),
        "f1: m7+sonnet vs m7+opus": ("f1", panels[("m7_fm_llm_gated", "claude-sonnet-4.6")],
                                     m["m7_fm_llm_gated"]),
        "fir: m6 vs m2": ("mean_fir", m["m6_fm_llm"], m["m2_snapshot_llm"]),
        "judge: m6 vs m2 (opus4.8)": ("mean_score", judges[("m6", "opus4.8")],
                                      judges[("m2", "opus4.8")]),
        "judge: m7 vs m6 (opus4.8)": ("mean_score", judges[("m7", "opus4.8")],
                                      judges[("m6", "opus4.8")]),
        "judge: m7+sonnet vs m7+opus": ("mean_score", judges[("m7+sonnet", "opus4.8")],
                                        judges[("m7", "opus4.8")]),
        "judge: m6 vs m2 (gemini)": ("mean_score", judges[("m6", "gemini")],
                                     judges[("m2", "gemini")]),
        "judge: m6 vs m2 (gpt5.5)": ("mean_score", judges[("m6", "gpt5.5")],
                                     judges[("m2", "gpt5.5")]),
    }
    print()
    for name, (kind, da, db) in tests.items():
        if kind == "f1":
            pa, pb = aligned(da, db, "precision", "precision")
            ra, rb = aligned(da, db, "recall", "recall")
            p = paired_perm_f1(pa, ra, pb, rb)
        elif kind == "mean_fir":
            a, b = aligned(da, db, "fir", "fir")
            p = paired_perm_test(a, b)
        else:
            a, b = aligned(da, db, "score", "score")
            p = paired_perm_test(a, b)
        out["tests"][name] = p
        print(f"perm  {name:34s} p = {p:.4f}")

    # ---- m1 rules baseline, if the local re-run has landed ----
    m1 = load_m1_weekly()
    if m1 is not None:
        rm, rlo, rhi = boot_ci_mean(m1["recall"])
        out["ci"]["m1/recall"] = [rm, rlo, rhi]
        # Micro precision: pool per-ticket hits, bootstrap over weeks.
        hits = m1["precision_hits"]
        n = len(hits)
        point = float(np.mean([h for w in hits for h in w])) if any(hits) else 0.0
        f1_point = f1_from_pr(point, rm)
        boots = []
        for _ in range(N_BOOT):
            idx = RNG.integers(0, n, size=n)
            ph = [h for i in idx for h in hits[i]]
            rw = m1["recall"][idx]
            rw = rw[~np.isnan(rw)]
            boots.append(f1_from_pr(float(np.mean(ph)) if ph else 0.0,
                                    float(np.mean(rw)) if len(rw) else 0.0))
        out["ci"]["m1/f1"] = [f1_point, float(np.percentile(boots, 2.5)),
                              float(np.percentile(boots, 97.5))]
        print(f"\nm1_persistence_rules     F1 {f1_point:.4f} "
              f"[{out['ci']['m1/f1'][1]:.4f},{out['ci']['m1/f1'][2]:.4f}]")
    else:
        print("\n(m1 weekly dump not present yet — skipped)")

    with open(RESULTS / "bootstrap_stats.json", "w") as f:
        json.dump(out, f, indent=1)
    print(f"\nwritten: {RESULTS / 'bootstrap_stats.json'}")


if __name__ == "__main__":
    main()

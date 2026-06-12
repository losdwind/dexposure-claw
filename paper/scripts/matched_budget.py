#!/usr/bin/env python3
"""Matched-budget recall@k / precision@k for the EMNLP paper.

Addresses the ticket-count confound: methods emit different numbers of
ticket targets per week, and unbudgeted recall mechanically favours
methods that flag more protocols. Here every method is truncated to its
top-k targets per week (LLM variants ranked by the LLM's own risk_score;
the m1 rules baseline ranked by ticket confidence), and recall@k /
precision@k are computed against the same truly-stressed pool.

Inputs (already on disk):
  results/run_20260515_pipeline_full/llm_eval/raw_m{2,6,7}_*.json
  results/redecide_20260603_023030/decide_*__claude-sonnet-4.6.json
  results/local_rerun_m1/b5_weekly__m1_persistence_rules.json  (optional)

Output: results/matched_budget.json + stdout table.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np

RESULTS = Path(__file__).resolve().parent.parent / "results"
RNG = np.random.default_rng(20260610)
KS = (1, 2, 3)
N_PERM = 20_000


def norm(name: str) -> str:
    """Same normalisation as llm_eval_b5.assess_week: strip a trailing
    '(Category)' suffix so name formatting does not bias matching."""
    if not isinstance(name, str):
        return ""
    return re.sub(r"\s*\([^)]*\)\s*$", "", name.strip()).strip()


def norm_ranked(names: list[str]) -> list[str]:
    seen, out = set(), []
    for n in names:
        n = norm(n)
        if n and n not in seen:
            seen.add(n)
            out.append(n)
    return out


def weeks_from_llm_raw(path: Path) -> list[dict]:
    raw = json.load(open(path))
    weeks = []
    for e in raw:
        primary = e["llm_outputs"][0]
        ranked = sorted(primary.get("target_protocols", []),
                        key=lambda t: -float(t.get("risk_score", 0.0)))
        weeks.append({
            "date": e["date"],
            "ranked_targets": norm_ranked([t["protocol"] for t in ranked]),
            "stressed": {norm(p) for p in e["truly_stressed"]} - {""},
        })
    return weeks


def weeks_from_m1(path: Path) -> list[dict]:
    d = json.load(open(path))
    weeks = []
    for w in d["weeks"]:
        ranked = [proto for proto, _score in w["targets_scored"]]
        weeks.append({
            "date": w["date"],
            "ranked_targets": norm_ranked(ranked),
            "stressed": {norm(p) for p in w["truly_stressed"]} - {""},
        })
    return weeks


def at_k(weeks: list[dict], k: int) -> dict:
    """Per-week recall@k and precision@k arrays (weeks with no targets count
    as 0 hits at every k; weeks with empty stress pool are skipped)."""
    rec, prec, n_used = [], [], []
    for w in weeks:
        if not w["stressed"]:
            continue
        top = w["ranked_targets"][:k]
        hits = sum(1 for t in top if t in w["stressed"])
        rec.append(hits / len(w["stressed"]))
        prec.append(hits / len(top) if top else 0.0)
        n_used.append(len(top))
    return {"recall": np.array(rec), "precision": np.array(prec),
            "mean_targets_used": float(np.mean(n_used)) if n_used else 0.0}


def paired_perm(a: np.ndarray, b: np.ndarray) -> float:
    d = a - b
    obs = abs(d.mean())
    signs = RNG.choice([-1.0, 1.0], size=(N_PERM, len(d)))
    null = np.abs((signs * d).mean(axis=1))
    return float((np.sum(null >= obs) + 1) / (N_PERM + 1))


def main() -> None:
    lle = RESULTS / "run_20260515_pipeline_full" / "llm_eval"
    rdir = RESULTS / "redecide_20260603_023030"
    methods: dict[str, list[dict]] = {
        "m2 (LLM, no FM)": weeks_from_llm_raw(lle / "raw_m2_snapshot_llm.json"),
        "m6 (FM+LLM)": weeks_from_llm_raw(lle / "raw_m6_fm_llm.json"),
        "m7 (FM+LLM+gate)": weeks_from_llm_raw(lle / "raw_m7_fm_llm_gated.json"),
        "m7 x Sonnet-4.6": [
            {"date": e["date"],
             "ranked_targets": norm_ranked([t["protocol"] for t in sorted(
                 e["llm_outputs"][0].get("target_protocols", []),
                 key=lambda t: -float(t.get("risk_score", 0.0)))]),
             "stressed": set()}  # stressed filled below from m7 raw (same weeks)
            for e in json.load(open(rdir / "decide_m7_fm_llm_gated__claude-sonnet-4.6.json"))["entries"]
        ],
    }
    # redecide entries don't store the stressed list; reuse m7's per-date pools.
    pool_by_date = {w["date"]: w["stressed"] for w in methods["m7 (FM+LLM+gate)"]}
    for w in methods["m7 x Sonnet-4.6"]:
        w["stressed"] = pool_by_date.get(w["date"], set())

    m1_path = RESULTS / "local_rerun_m1" / "b5_weekly__m1_persistence_rules.json"
    if m1_path.exists():
        methods = {"m1 (persist.+rules)": weeks_from_m1(m1_path), **methods}
    else:
        print("(m1 weekly dump not present — m1 rows skipped)\n")

    out: dict = {"ks": list(KS), "rows": {}, "tests": {}}
    print(f"{'method':22s}" + "".join(f"  R@{k}x1e3  P@{k}  " for k in KS))
    per_method_rec = {}
    for name, weeks in methods.items():
        cells = []
        for k in KS:
            r = at_k(weeks, k)
            per_method_rec[(name, k)] = (r["recall"], [w["date"] for w in weeks if w["stressed"]])
            cells.append((float(r["recall"].mean()), float(r["precision"].mean())))
        out["rows"][name] = cells
        print(f"{name:22s}" + "".join(f"  {rm*1000:6.2f}  {pm:5.3f} " for rm, pm in cells))

    # Paired tests at matched k (align by date).
    def get(name, k):
        rec, dates = per_method_rec[(name, k)]
        return dict(zip(dates, rec))

    pairs = [("m6 (FM+LLM)", "m2 (LLM, no FM)"),
             ("m7 (FM+LLM+gate)", "m2 (LLM, no FM)")]
    if m1_path.exists():
        pairs += [("m7 (FM+LLM+gate)", "m1 (persist.+rules)"),
                  ("m6 (FM+LLM)", "m1 (persist.+rules)")]
    print()
    for a_name, b_name in pairs:
        for k in KS:
            da, db = get(a_name, k), get(b_name, k)
            common = sorted(set(da) & set(db))
            a = np.array([da[c] for c in common])
            b = np.array([db[c] for c in common])
            p = paired_perm(a, b)
            key = f"recall@{k}: {a_name} vs {b_name}"
            out["tests"][key] = {"p": p, "mean_a": float(a.mean()),
                                 "mean_b": float(b.mean()), "n": len(common)}
            print(f"perm  {key:55s} {a.mean()*1000:5.2f} vs {b.mean()*1000:5.2f} (x1e3)  p={p:.4f}")

    with open(RESULTS / "matched_budget.json", "w") as f:
        json.dump(out, f, indent=1)
    print(f"\nwritten: {RESULTS / 'matched_budget.json'}")


if __name__ == "__main__":
    main()

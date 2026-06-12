#!/usr/bin/env python3
"""Build leakage-controlled (masked) versions of the exported crisis prompts.

Motivation: the decision LLM's pretraining covers Terra/FTX/SVB. In the
unmasked replay 65% of weekly rationales explicitly name historical events,
so those numbers are upper bounds. This script removes every channel that
lets a model look up the answer instead of reading the evidence:

  1. Absolute dates -> relative ("week T"); regex-assert no year survives.
  2. ALL protocol identifiers (numeric DefiLlama-style IDs *and* token
     symbols like USDC/LUNC) -> randomly permuted synthetic IDs (P####),
     one consistent mapping per crisis window. truly_stressed is mapped
     with the same table so scoring and judging stay in one namespace.
  3. Dollar magnitudes -> rescaled by a per-window random factor in
     [0.3, 3.0] and rounded to 3 significant figures (kills exact-value
     fingerprints while preserving relative structure).
  4. Node/edge counts -> 2 significant figures.
  5. Residual sweep: assert no event-identifying token remains.

Output: paper/results/crisis_masked_raw/<window>/raw_<method>.json with the
same schema redecide_b5.py expects. Mapping tables are saved alongside for
un-masking analyses later.
"""
from __future__ import annotations

import json
import random
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RAW_BASE = ROOT / "paper/results/crisis_pre2022_raw/runs/20260612_073921/logs/crisis_backtest"
OUT_BASE = ROOT / "paper/results/crisis_masked_raw"

WINDOWS = {
    "terra": "llm_eval_20260612_073443",
    "ftx": "llm_eval_20260612_073555",
    "svb": "llm_eval_20260612_073719",
}
METHODS = ["m6_fm_llm", "m7_fm_llm_gated"]

# Lines like "  1. USDC (Unknown) -- weight: ..." / "    - 1571 (Cross Chain): loss=..."
IDENT_RE = re.compile(r"(?m)^\s*(?:\d+\.|-)\s+(.+?) \([^)]*\)")
LEAK_RE = re.compile(
    r"20\d\d-\d\d-\d\d|terra|luna|lunc|\bust\b|anchor|ftx|alameda|svb|silicon valley|"
    r"celsius|voyager|3ac|usdc|usdt|weth|wbtc|\bdai\b|busd|tusd|frax|steth",
    re.I,
)


def sig_round(x: float, sig: int = 3) -> float:
    if x == 0:
        return 0.0
    from math import floor, log10
    return round(x, -int(floor(log10(abs(x)))) + (sig - 1))


def mask_window(window: str, eval_dir: str) -> None:
    rng = random.Random(f"mask-{window}")
    scale = rng.uniform(0.3, 3.0)
    out_dir = OUT_BASE / window
    out_dir.mkdir(parents=True, exist_ok=True)

    # Collect every identifier across both methods first -> one mapping.
    raws = {}
    idents: set[str] = set()
    for m in METHODS:
        raws[m] = json.load(open(RAW_BASE / eval_dir / f"raw_{m}.json"))
        for e in raws[m]:
            idents.update(s.strip() for s in IDENT_RE.findall(e["user_prompt"]))
            idents.update(str(p).strip() for p in e["truly_stressed"])
    idents.discard("")
    pool = rng.sample(range(1000, 9999), len(idents))
    mapping = {ident: f"P{pool[i]}" for i, ident in enumerate(sorted(idents))}

    # Replace identifiers only in identifier positions (list-rank numbers
    # like "2." and the "week T+2" index must survive untouched).
    def sub_idents(text: str) -> str:
        def repl(mo: re.Match) -> str:
            rel_s = mo.start(1) - mo.start(0)
            rel_e = mo.end(1) - mo.start(0)
            g = mo.group(0)
            return g[:rel_s] + mapping.get(mo.group(1).strip(), mo.group(1)) + g[rel_e:]
        return IDENT_RE.sub(repl, text)

    def rescale(mo: re.Match) -> str:
        if mo.group(3):  # percentage -> relative, keep as-is
            return mo.group(0)
        return f"{mo.group(1)}{sig_round(float(mo.group(2)) * scale):.2f}"

    n_weeks_total = 0
    for m in METHODS:
        masked = []
        for idx, e in enumerate(raws[m]):
            up = e["user_prompt"]
            up = re.sub(
                r"Current DeFi network analysis \([0-9-]+\)",
                f"Current DeFi network analysis (week T{idx:+d})".replace("+0", ""),
                up,
            )
            up = re.sub(r"(weight: |loss=)(\d+(?:\.\d+)?)(%?)", rescale, up)
            up = re.sub(
                r"Nodes: (\d+) protocols, Edges: (\d+)",
                lambda mo: f"Nodes: ~{sig_round(int(mo.group(1)), 2):.0f} protocols, "
                           f"Edges: ~{sig_round(int(mo.group(2)), 2):.0f}",
                up,
            )
            up = sub_idents(up)
            leftover = LEAK_RE.findall(up)
            if leftover:
                sys.exit(f"FATAL leak in {window}/{m}/{e['date']}: {set(leftover)}")
            masked.append({
                **e,
                "user_prompt": up,
                "truly_stressed": sorted({
                    mapping.get(str(p).strip(), str(p)) for p in e["truly_stressed"]
                }),
            })
            n_weeks_total += 1
        json.dump(masked, open(out_dir / f"raw_{m}.json", "w"), indent=1)
    json.dump({"scale": scale, "mapping": mapping},
              open(out_dir / "mask_mapping.json", "w"), indent=1)
    print(f"{window}: {n_weeks_total} entries masked, "
          f"{len(mapping)} identifiers, scale={scale:.3f}")


if __name__ == "__main__":
    for w, d in WINDOWS.items():
        mask_window(w, d)
    print("ALL WINDOWS MASKED ->", OUT_BASE)

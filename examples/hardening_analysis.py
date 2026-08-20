"""
Summarize gate saturation and soft-vs-hardened agreement across discovery seeds.

Usage:  python hardening_analysis.py path/to/runs.jsonl
"""

import json
import sys
from collections import Counter

import numpy as np

path = sys.argv[1]

last_n = int(sys.argv[2]) if len(sys.argv) > 2 else None
rows = [json.loads(line) for line in open(path) if line.strip()]
if last_n:
    rows = rows[-last_n:]

print(f"total rows: {len(rows)}")
print("schemas present:")
for keys, count in Counter(tuple(sorted(r.keys())) for r in rows).items():
    print(f"  n={count}  {list(keys)}")
print()


def col(rows, key):
    """Values for `key`, skipping rows that lack it or hold null/non-finite."""
    out = []
    for r in rows:
        v = r.get(key)
        if v is None:
            continue
        v = float(v)
        if np.isfinite(v):
            out.append(v)
    return np.array(out)


def describe(name, v):
    if len(v) == 0:
        print(f"{name:<22} (absent)")
        return
    print(
        f"{name:<22} n={len(v):<4} median={np.median(v):.4g}  "
        f"IQR=[{np.percentile(v, 25):.4g}, {np.percentile(v, 75):.4g}]  "
        f"min={v.min():.4g}  max={v.max():.4g}"
    )


# --- restrict to the discovery runs, if the file mixes conditions ---
if any("init" in r for r in rows):
    print("init values:", Counter(r.get("init", "unlabeled") for r in rows))
    print()

emerged = [r for r in rows if r.get("emerged")]
print(f"emerged: {len(emerged)}/{len(rows)}")
print()

# --- the two quantities the hardening claim rests on ---
print("ALL RUNS")
describe("soft_mse", col(rows, "soft_mse"))
describe("hard_dyn_divergence", col(rows, "hard_dyn_divergence"))
describe("frac_committed", col(rows, "frac_committed"))
describe("frac_ambiguous", col(rows, "frac_ambiguous"))
describe("mean_dist_mid", col(rows, "mean_dist_mid"))
print()

print("EMERGED ONLY")
describe("soft_mse", col(emerged, "soft_mse"))
describe("hard_dyn_divergence", col(emerged, "hard_dyn_divergence"))
describe("frac_committed", col(emerged, "frac_committed"))
describe("frac_ambiguous", col(emerged, "frac_ambiguous"))
print()

# --- does divergence track commitment? the mechanism claim ---
d = col(emerged, "hard_dyn_divergence")
c = col(emerged, "frac_committed")
if len(d) == len(c) and len(d) > 2:
    print(f"corr(hard_dyn_divergence, frac_committed) = {np.corrcoef(d, c)[0, 1]:+.3f}")

# --- how many hardened cleanly? pick a threshold and report the fraction ---
if len(d):
    for tol in (1e-4, 1e-3, 1e-2, 1e-1):
        print(f"  hard_dyn_divergence < {tol:g}: {int((d < tol).sum())}/{len(d)}")

# --- gate distribution, if the raw u values were saved alongside ---
try:
    u = np.load(path.replace(".jsonl", "_u.npz"))["u"]
    print()
    print(f"gate values: n={len(u)}")
    print(f"  in [0.0,0.1): {np.mean(u < 0.1):.3f}")
    print(f"  in [0.1,0.9): {np.mean((u >= 0.1) & (u <= 0.9)):.3f}")
    print(f"  in (0.9,1.0]: {np.mean(u > 0.9):.3f}")
    print(f"  within 0.05 of 0.5: {np.mean(np.abs(u - 0.5) < 0.05):.3f}")
except (FileNotFoundError, KeyError):
    pass

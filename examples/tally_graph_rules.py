"""
tally_graph_rules.py

Aggregates graph_rules_check.py + the flatness cross-check into counts by
measured_support_size, so the pattern is exact rather than eyeballed from
150 printed rows.
"""

from pathlib import Path
from collections import defaultdict

import numpy as np

from graph_rules_check import analyze_graph_rules, measured_fixedpoint_support
from diagnose_boundedness import hard_len1_is_genuine_fixedpoint
from support_sequence_analysis import parse_seed_zstd

results_dir = Path("results")
param_files = sorted(results_dir.glob("params_seed*_z*.npy"))

by_size = defaultdict(list)

for pf in param_files:
    seed, z_std = parse_seed_zstd(pf)
    gr = analyze_graph_rules(pf)
    support = measured_fixedpoint_support(pf)
    size = len(support)

    is_flat = None
    if size < 5:
        fp_check = hard_len1_is_genuine_fixedpoint(pf)
        is_flat = max(fp_check["tail_range_rel_to_scale"]) < 1e-3

    by_size[size].append({
        "seed": seed, "z_std": z_std,
        "n_stable_fps": gr["n_stable_fixed_points"],
        "flat": is_flat,
    })

print(f"{'size':>4} | {'n':>3} | {'flat=True':>9} | {'flat=False':>10} | "
      f"{'n_stable_fps=0 (among flat=False)':>34}")
for size in sorted(by_size, reverse=True):
    rows = by_size[size]
    n = len(rows)
    n_flat_true = sum(r["flat"] is True for r in rows)
    n_flat_false = sum(r["flat"] is False for r in rows)
    n_flat_false_zero_stable = sum(
        r["flat"] is False and r["n_stable_fps"] == 0 for r in rows
    )
    print(f"{size:>4} | {n:>3} | {n_flat_true:>9} | {n_flat_false:>10} | {n_flat_false_zero_stable:>34}")

# The interesting mismatch: support < 5, theory predicts a stable FP on
# THAT exact support, but numerics say it's not flat (still moving).
print("\nPartial mismatches (theory's predicted stable support size matches\n"
      "measured support size, but numerically NOT flat):")
for size, rows in by_size.items():
    if size == 5:
        continue
    for r in rows:
        if r["flat"] is False and r["n_stable_fps"] >= 1:
            print(f"  seed {r['seed']:>3} z={r['z_std']:.1f}  support_size={size} "
                  f"n_stable_fps={r['n_stable_fps']}")

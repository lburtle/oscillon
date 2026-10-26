"""
check_task_fidelity.py

Answers the actual question: after hardening, is the network still usable
for the learned task (tracking the target sinusoids), or does it merely
keep oscillating structurally while the readout accuracy collapses?

Pulls hard_mse / soft_mse / hard_dyn_divergence from the CORRECTED sweep
output (results/uniform_pop512.json, post bug-fix) and splits by the three
structural categories already established (full-support oscillation /
true fixed-point collapse / sub-cycle collapse).
"""

import json
from pathlib import Path
from collections import defaultdict

import numpy as np

from graph_rules_check import measured_fixedpoint_support
from diagnose_boundedness import hard_len1_is_genuine_fixedpoint
from support_sequence_analysis import parse_seed_zstd

# --- load the CORRECTED sweep's own recorded soft_mse/hard_mse per (seed, z_std) ---
with open("results/uniform_pop512.json") as f:
    sweep_records = json.load(f)

by_key = {(r["seed"], r["z_std"]): r for r in sweep_records}

results_dir = Path("results")  # saved params, unaffected by the bug
param_files = sorted(results_dir.glob("params_seed*_z*.npy"))

by_category = defaultdict(list)

for pf in param_files:
    seed, z_std = parse_seed_zstd(pf)
    rec = by_key.get((seed, z_std))
    if rec is None:
        print(f"WARNING: no corrected-sweep record for seed {seed} z={z_std} -- skipping")
        continue

    support = measured_fixedpoint_support(pf)
    if len(support) == 5:
        category = "full_support_oscillation"
    else:
        fp_check = hard_len1_is_genuine_fixedpoint(pf)
        is_flat = max(fp_check["tail_range_rel_to_scale"]) < 1e-3
        category = "true_fixedpoint_collapse" if is_flat else "subcycle_collapse"

    by_category[category].append({
        "seed": seed, "z_std": z_std,
        "soft_mse": rec["soft_mse"],
        "hard_mse": rec["hard_mse"],
        "hard_dyn_divergence": rec["hard_dyn_divergence"],
    })

print(f"{'category':>28} | {'n':>3} | {'median soft_mse':>16} | "
      f"{'median hard_mse':>16} | {'median divergence':>18} | {'hard/soft ratio':>15}")
for cat, rows in by_category.items():
    n = len(rows)
    soft_mse = np.median([r["soft_mse"] for r in rows])
    hard_mse = np.median([r["hard_mse"] for r in rows])
    div = np.median([r["hard_dyn_divergence"] for r in rows])
    ratio = hard_mse / soft_mse if soft_mse > 0 else float("inf")
    print(f"{cat:>28} | {n:>3} | {soft_mse:>16.4f} | {hard_mse:>16.4f} | "
          f"{div:>18.4f} | {ratio:>15.2f}")

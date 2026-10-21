"""
check_pairing_structure.py

Checks whether the "0&1 co-fire, 2&3 co-fire" pattern seen in the sustained-
oscillation figure example (seed 0, z=2.0) is explained by graph structure:
in this CTLN weight convention, an edge (i,j) is WEAK inhibition (~-1+eps,
allows co-activation) and a non-edge is STRONG inhibition (~-1-delta,
enforces competition/alternation). Two canonical weight values should appear
in W_hard; this reports which neuron pairs get the "weak" (edge, can
co-fire) value and which get the "strong" (non-edge, competes) value,
directly testing the co-firing hypothesis against the actual discovered
graph rather than inferring it from the raster alone.

Standalone / non-invasive: reads existing params only, writes nothing.
"""

from pathlib import Path

import numpy as np
import jax.numpy as jnp

from support_sequence_analysis import hardened_weights, spec, parse_seed_zstd

RESULTS_DIR = Path("results_before_bug_fix")  # adjust to match your current results dir
SEED, Z_STD = 14, 2.0  # the sustained-oscillation figure example


if __name__ == "__main__":
    pf = next(RESULTS_DIR.glob(f"params_seed{SEED}_z{Z_STD}.npy"), None)
    if pf is None:
        raise FileNotFoundError(f"no params file for seed {SEED} z={Z_STD} in {RESULTS_DIR}")

    params = jnp.asarray(np.load(pf))
    W_hard, theta_hard = hardened_weights(spec, params)
    W = np.asarray(W_hard)
    n = W.shape[0]

    off_diag = W[~np.eye(n, dtype=bool)]
    weak_val = off_diag.max()   # least negative -- edge (weak inhibition, can co-fire)
    strong_val = off_diag.min()  # most negative -- non-edge (strong inhibition, competes)

    print(f"seed {SEED} z={Z_STD}")
    print(f"canonical weak (edge) value:   {weak_val:.4f}")
    print(f"canonical strong (non-edge) value: {strong_val:.4f}\n")

    print("Full weight matrix W_hard[i,j] (row i -> col j):")
    header = "      " + "".join(f"{j:>9}" for j in range(n))
    print(header)
    for i in range(n):
        row = "".join(f"{W[i, j]:>9.4f}" if i != j else f"{'--':>9}" for j in range(n))
        print(f"  {i:>2} |{row}")

    print("\nPairwise relationship (edge = weak = can co-fire; non-edge = strong = competes):")
    for i in range(n):
        for j in range(i + 1, n):
            ij, ji = W[i, j], W[j, i]
            ij_type = "edge (weak)" if abs(ij - weak_val) < abs(ij - strong_val) else "non-edge (strong)"
            ji_type = "edge (weak)" if abs(ji - weak_val) < abs(ji - strong_val) else "non-edge (strong)"
            flag = "  <-- both directions weak: no competition, CAN co-fire" if (
                ij_type.startswith("edge") and ji_type.startswith("edge")
            ) else ""
            print(f"  ({i},{j}): W[{i},{j}]={ij:.4f} ({ij_type})  "
                  f"W[{j},{i}]={ji:.4f} ({ji_type}){flag}")

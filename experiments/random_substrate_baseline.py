"""
random_substrate_baseline.py

Random (untrained) substrate + closed-form readout baseline, same protocol as the main sweep:
N=5, dt=0.1, 702 steps, burn-in 200, ridge 1e-6, eps=0.10, delta=0.50, uniform theta, x0=0.1.

Question answered: how much of the readout fit is the readout's capacity on ANY oscillating network, and how much
needs the trained substrate? No ES, no training: draw a substrate, simulate it, fit the readout, record the MSE.

Substrate families (all inside the CTLN weight band, so the comparison is like for like):
  soft_init    Z ~ N(0, sigma_z^2) (z in 1,2,3), W = -1-delta + (eps+delta)*sigmoid(Z): exactly the ES starting point,
               untrained. (The paper states these never oscillate, so this is "readout on a non-oscillating net".)
  digraph_p    random directed graph, each off-diagonal edge present with prob p, hardened band endpoints (a genuine
               CTLN). Most do not oscillate; we record all of them AND, separately, rejection-sample until
               N_OSC oscillating ones are found (the strongest random baseline: oscillation by luck).

Two targets per network:
  fixed   period 2*pi, as every unstructured trained run is fit against (the like-for-like comparison)
  own     period = the network's own measured period (only for oscillating nets): the most favorable case for the
          baseline. If random oscillating nets fit well even here, readout capacity explains much of the result.

Usage (from experiments/, next to topo_ablation_sweep.py):
  python random_substrate_baseline.py                 # runs everything, writes results_random_baseline/records.json
  python random_substrate_baseline.py --summary       # prints the summary table from the saved records
"""
import argparse
import json
from pathlib import Path

import jax.numpy as jnp
import numpy as np

from oscillon.ctln.dynamics import simulate
from oscillon.ctln.readout import apply_readout, fit_readout
from topo_ablation_sweep import (BURN_IN, DELTA, DT, EPS, N, N_STEPS, REF_PERIOD, RIDGE, THETA_INIT, X0,
                                 cycle_emerged, make_targets, measure_period)

OUT = Path(__file__).parent / "results_random_baseline"
N_SOFT = 50                      # soft-init nets per sigma_z
N_ALL = 300                      # random digraphs per density, oscillating or not
N_OSC = 50                       # oscillating digraphs to collect per density (rejection sampling)
MAX_TRIES = 20000
DENSITIES = (0.25, 0.5)          # edge probability; pick to bracket the density of your trained graphs
ZS = (1.0, 2.0, 3.0)

T_GRID = jnp.arange(N_STEPS) * DT
THETA = jnp.full((N,), THETA_INIT)   # if simulate() wants a scalar theta, change this one line to jnp.asarray(THETA_INIT)


def soft_W(rng, z):
    Z = rng.normal(0.0, z, size=(N, N))
    g = 1.0 / (1.0 + np.exp(-Z))
    W = -1.0 - DELTA + (EPS + DELTA) * g
    np.fill_diagonal(W, 0.0)
    return W


def digraph_W(rng, p):
    A = rng.random((N, N)) < p                       # A[i, j]: edge j -> i
    np.fill_diagonal(A, False)
    W = np.where(A, -1.0 + EPS, -1.0 - DELTA)
    np.fill_diagonal(W, 0.0)
    return W, A


def fit_mse(xs_full, targets):
    xs = jnp.asarray(xs_full)[BURN_IN:]
    tgt = targets[BURN_IN:]
    m = min(xs.shape[0], tgt.shape[0])
    xs, tgt = xs[:m], tgt[:m]
    R, b = fit_readout(xs, tgt, ridge=RIDGE)
    mse = float(jnp.mean((apply_readout(xs, R, b) - tgt) ** 2))
    const = float(jnp.mean((tgt.mean(0) - tgt) ** 2))
    return mse, const


def evaluate(W, **meta):
    Wj = jnp.asarray(W)
    xs_full = np.asarray(simulate(Wj, THETA, X0, dt=DT, n_steps=N_STEPS))
    emerged, amp, per = cycle_emerged(Wj, THETA, X0, DT, N_STEPS, BURN_IN)
    rec = dict(meta, emerged=bool(emerged), amp=float(amp), period=per, finite=bool(np.isfinite(xs_full).all()))
    if not rec["finite"]:
        return rec
    mse, const = fit_mse(xs_full, make_targets(N, T_GRID, REF_PERIOD))
    rec.update(mse_fixed=mse, norm_fixed=mse / const)
    if emerged and per is not None and per * 1.0 < (N_STEPS - BURN_IN) * DT / 2:     # need >= 2 periods in the fit window
        mse_o, const_o = fit_mse(xs_full, make_targets(N, T_GRID, per))
        rec.update(mse_own=mse_o, norm_own=mse_o / const_o)
    return rec


def run_all():
    OUT.mkdir(exist_ok=True)
    recs = []
    rng = np.random.default_rng(12345)

    for z in ZS:                                      # soft init, untrained
        for s in range(N_SOFT):
            recs.append(evaluate(soft_W(rng, z), family=f"soft_init_z{z:.0f}", seed=s))

    for p in DENSITIES:
        for s in range(N_ALL):                        # all random digraphs, oscillating or not
            W, A = digraph_W(rng, p)
            recs.append(evaluate(W, family=f"digraph_p{p}_all", seed=s, edges=int(A.sum())))
        found, tries = 0, 0                           # oscillating only (rejection sampling)
        while found < N_OSC and tries < MAX_TRIES:
            tries += 1
            W, A = digraph_W(rng, p)
            rec = evaluate(W, family=f"digraph_p{p}_osc", seed=found, edges=int(A.sum()))
            if rec["emerged"]:
                rec["tries_so_far"] = tries
                recs.append(rec)
                found += 1
        print(f"p={p}: {found} oscillating digraphs found in {tries} tries")

    (OUT / "records.json").write_text(json.dumps(recs, indent=1))
    return recs


def summary(recs):
    def stats(v):
        v = np.asarray(v, float)
        if len(v) == 0:
            return "n=0"
        q = np.percentile(v, [25, 50, 75])
        return f"n={len(v):3d} median={q[1]:.4f} IQR=[{q[0]:.4f}, {q[2]:.4f}]"

    for fam in sorted({r["family"] for r in recs}):
        rs = [r for r in recs if r["family"] == fam]
        em = [r for r in rs if r["emerged"]]
        print(f"\n{fam}: {len(rs)} nets, oscillating {len(em)}, non-finite {sum(not r['finite'] for r in rs)}")
        print("  normalized MSE, 2pi target, all finite :", stats([r["norm_fixed"] for r in rs if "norm_fixed" in r]))
        print("  normalized MSE, 2pi target, oscillating:", stats([r["norm_fixed"] for r in em if "norm_fixed" in r]))
        print("  normalized MSE, own-period target      :", stats([r["norm_own"] for r in em if "norm_own" in r]))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--summary", action="store_true")
    args = ap.parse_args()
    recs = json.loads((OUT / "records.json").read_text()) if args.summary else run_all()
    summary(recs)

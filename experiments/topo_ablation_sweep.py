"""
topo_ablation_sweep.py

Fixed-topology ablation, rewritten as a standalone sweep (the original script's module-level demo is dropped).

What it fixes relative to the old sweep inside the topology-ablation script:
  1. The z_std loop did nothing: the init ignored z_std and the key depended only on the seed, so the three "z_std"
     conditions were the same run three times. Here INIT=asym (default) draws the initial logits exactly like the main
     sweep (init_params_asymmetric, same key derivation), restricted by the mask, so z_std matters and the runs are
     paired with the main sweep. INIT=seed reproduces the old init (init_block_params_from_adjacency) and runs once
     per seed (z_std is meaningless there).
  2. Adjacency is read from the simulated weights W (threshold at the band midpoint), not from raw logits (masked
     logits drift freely and give phantom edges).
  3. The hardened-CTLN check and the edge-commitment statistics are dropped: both read raw logits and ignore the mask,
     so they are meaningless for a fixed topology.
  4. Nothing is appended to old files and no paper figure is overwritten: each run writes its own record file into a
     NEW output directory (default results_topo_abl_v2/), existing records are skipped (resumable), and a --merge
     step builds the summary. Runs for different seeds or z_std can go in separate Slurm jobs.

Usage (run from the experiments/ folder, next to topology_analysis.py):
  python topo_ablation_sweep.py --seeds 0-49 --zs 1.0,2.0,3.0          # everything, sequentially
  python topo_ablation_sweep.py --seeds 0-49 --zs 2.0                  # one z_std per job
  python topo_ablation_sweep.py --merge                                # summary of all finished records
Env: INIT=asym|seed (default asym); OUT=<dir> (default results_topo_abl_v2).
"""
import argparse
import json
import os
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from oscillon.ctln.dynamics import simulate
from oscillon.ctln.graph import MotifSpec
from oscillon.ctln.readout import apply_readout, fit_readout
from oscillon.ctln.topology import (
    BlockSoftSpec,
    block_gate_matrix,
    init_block_params_from_adjacency,
    init_params_asymmetric,
    make_fixed_topology_p2m,
)
from oscillon.search.train import train
from topology_analysis import analyze_interior_fixed_point

# ---------------------------------------------------------------- protocol (matches the main sweep) ----
N = 5
DT, BURN_IN = 0.1, 200
THETA_INIT = 0.5
EPS, DELTA = 0.10, 0.50
RIDGE = 1e-6
REF_PERIOD = 2.0 * np.pi                                  # fixed reference target, as in the main sweep
N_STEPS = int(BURN_IN + 8 * (REF_PERIOD / DT))            # 702
WARM = dict(n_iters=9000, pop=512, sigma=2, lr=0.05)
CRYST = dict(n_iters=5000, pop=512, sigma=1, lr=0.03)
MID = -1.0 - DELTA + (EPS + DELTA) / 2.0                  # -1.2: threshold between edge and non-edge weight

INIT = os.environ.get("INIT", "asym")
MASK = os.environ.get("MASK", "chain")      # chain: 0->1->2->3->4 (acyclic); cycle: chain plus 4->0 (directed 5-cycle)
assert MASK in ("chain", "cycle")
OUT = Path(os.environ.get("OUT", Path(__file__).parent / ("results_topo_abl_v2" if MASK == "chain" else "results_topo_abl_cycle_v2")))

spec = BlockSoftSpec(sizes=(N,), cross_deltas=np.zeros((1, 1)), eps=EPS, delta=DELTA, theta_init=THETA_INIT)
X0 = jnp.full((N,), 0.1)

# 0 -> 1 -> 2 -> 3 -> 4 (acyclic chain); A[i+1, i] = edge i -> i+1
A_SEED = np.zeros((N, N), bool)
for i in range(N - 1):
    A_SEED[i + 1, i] = True
if MASK == "cycle":
    A_SEED[0, N - 1] = True                                  # close the chain: edge 4 -> 0
P2M = make_fixed_topology_p2m(spec, A_SEED)


# ---------------------------------------------------------------- helpers (copied from the original script) ----
def measure_period(xs, dt, rel_thresh=0.5):
    """Peak-to-peak period of neuron 0, in time units. None if no clear cycle."""
    x = np.asarray(xs[:, 0])
    hi = rel_thresh * x.max()
    peaks = np.where((x[1:-1] > x[:-2]) & (x[1:-1] > x[2:]) & (x[1:-1] > hi))[0]
    if len(peaks) < 2:
        return None
    return float(np.mean(np.diff(peaks))) * dt


def make_targets(n, t, period, amplitudes=None, phases=None, offset=0.0):
    omega = 2.0 * np.pi / period
    if phases is None:
        phases = [2.0 * np.pi * k / n for k in range(n)]
    if amplitudes is None:
        amplitudes = [1.0 + 0.5 * (k % 3) for k in range(n)]
    cols = [a * jnp.sin(omega * t + ph) + offset for a, ph in zip(amplitudes, phases, strict=True)]
    return jnp.stack(cols, axis=1)


def contains_directed_cycle(A, n):
    """Hamiltonian directed cycle (visits all n, returns to start); A[i, j] = edge j -> i."""
    import itertools

    for perm in itertools.permutations(range(1, n)):
        order = (0,) + perm
        if all(A[order[(k + 1) % n], order[k]] for k in range(n)):
            return True
    return False


def discovered_cycle_graph(A, n):
    """Exactly a directed n-cycle (any rotation)?"""
    canonical = MotifSpec.cyclic(n).adjacency
    for shift in range(n):
        rolled = np.roll(np.roll(canonical, shift, axis=0), shift, axis=1)
        if np.array_equal(A, rolled):
            return True
    return False


def cycle_emerged(W_t, theta_t, x0, dt, n_steps, burn_in, min_amp=0.05):
    xs = np.asarray(simulate(W_t, theta_t, x0, dt=dt, n_steps=n_steps))[burn_in:]
    tail = xs[-int(0.4 * len(xs)):]
    amp = float(tail.std(0).max())
    per = measure_period(xs, dt)
    return (amp > min_amp and per is not None), amp, per


# ---------------------------------------------------------------- one run ----
def run_one(seed, z):
    """z is the logit-init std (INIT=asym) or None (INIT=seed)."""
    t = jnp.arange(N_STEPS) * DT
    targets = make_targets(N, t, REF_PERIOD)

    key = jax.random.PRNGKey(1000 + seed)
    key, sub = jax.random.split(key)
    if INIT == "asym":
        params0 = init_params_asymmetric(spec, sub, z_std=z)
    else:
        params0 = init_block_params_from_adjacency(spec, sub, A_SEED)

    res = train(spec, params0, X0, targets, key, p2m=P2M, dt=DT, n_steps=N_STEPS, burn_in=BURN_IN,
                warm=WARM, cryst=CRYST)

    # readout error of the trained (soft) network, same protocol as the main sweep
    xs = jnp.asarray(simulate(res.W, res.theta, X0, dt=DT, n_steps=N_STEPS))[BURN_IN:]
    tgt = targets[BURN_IN:]
    m = min(xs.shape[0], tgt.shape[0])
    xs, tgt = xs[:m], tgt[:m]
    R, b = fit_readout(xs, tgt, ridge=RIDGE)
    soft_mse = float(jnp.mean((apply_readout(xs, R, b) - tgt) ** 2))
    const = float(jnp.mean((tgt.mean(0) - tgt) ** 2))

    emerged, amp, per = cycle_emerged(res.W, res.theta, X0, DT, N_STEPS, BURN_IN)

    # adjacency from the weights actually simulated (NOT from raw logits)
    A_W = np.asarray(res.W) > MID
    np.fill_diagonal(A_W, False)
    fp = analyze_interior_fixed_point(res.W, res.theta)
    G = np.asarray(block_gate_matrix(spec, res.params))

    return res, dict(
        seed=seed, z_std=z, init=INIT,
        emerged=bool(emerged), amp=amp, period=per,
        soft_mse=soft_mse, const_mse=const, norm_mse=soft_mse / const,
        final_reward=float(res.history[-1]),
        new_edges=int((A_W & ~A_SEED).sum()),           # must be 0 (the mask held)
        seed_edges_kept=int((A_W & A_SEED).sum()),      # of the mask edges (4 for chain, 5 for cycle)
        mask=MASK, mask_edges=int(A_SEED.sum()),
        acyclic=not contains_directed_cycle(A_W, N),    # must be True
        hamiltonian_cycle=bool(contains_directed_cycle(A_W, N)),
        cycle_topology=bool(discovered_cycle_graph(A_W, N)),
        fp_unstable=not fp["stable"], max_real_eig=fp["max_real_eig"],
        seed_edge_gate_mean=float(G[A_SEED].mean()),
    )


def rec_path(seed, z):
    tag = "seed" if z is None else f"{z:.1f}"
    return OUT / f"rec_seed{seed}_z{tag}.json"


def parse_seeds(s):
    if "-" in s and "," not in s:
        a, b = s.split("-")
        return list(range(int(a), int(b) + 1))
    return [int(x) for x in s.split(",")]


# ---------------------------------------------------------------- merge ----
def merge():
    recs = [json.load(open(p)) for p in sorted(OUT.glob("rec_seed*_z*.json"))]
    if not recs:
        print("no records in", OUT)
        return
    json.dump(recs, open(OUT / "topo_abl_v2.json", "w"), indent=2, default=float)
    n = len(recs)
    mse = np.array([r["soft_mse"] for r in recs])
    nm = np.array([r["norm_mse"] for r in recs])
    print("=" * 70)
    print(f"{n} records in {OUT}  (init = {sorted({r['init'] for r in recs})}, mask = {sorted({r.get('mask', 'chain') for r in recs})})")
    print(f"mask edges kept in full:   {sum(r['seed_edges_kept'] == r.get('mask_edges', 4) for r in recs)}/{n}")
    print(f"oscillation emerged:       {sum(r['emerged'] for r in recs)}/{n}")
    print(f"new edges formed (want 0): {sum(r['new_edges'] > 0 for r in recs)}/{n} runs with any")
    print(f"acyclic (want all):        {sum(r['acyclic'] for r in recs)}/{n}")
    print(f"hamiltonian (from W):      {sum(r['hamiltonian_cycle'] for r in recs)}/{n}")
    print(f"interior fixed point unstable: {sum(r['fp_unstable'] for r in recs)}/{n}")
    print(f"soft_mse  median {np.median(mse):.4f}  IQR [{np.percentile(mse, 25):.4f}, {np.percentile(mse, 75):.4f}]"
          f"   min {mse.min():.4f}  max {mse.max():.4f}")
    print(f"normalized median {np.median(nm):.4f}   (constant-mean const_mse median {np.median([r['const_mse'] for r in recs]):.4f})")
    by = {}
    for r in recs:
        by.setdefault(r["z_std"], []).append(r)
    for z in sorted(by, key=lambda v: (v is None, v)):
        rs = by[z]
        print(f"  z_std={z}: n={len(rs)}  emerged={sum(r['emerged'] for r in rs)}  "
              f"median soft_mse={np.median([r['soft_mse'] for r in rs]):.4f}")
    # sanity: are the runs actually distinct?
    print(f"distinct soft_mse values (rounded to 1e-8): {len(set(np.round(mse, 8)))} of {n}")
    print("=" * 70)


# ---------------------------------------------------------------- main ----
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default="0-49")
    ap.add_argument("--zs", default="1.0,2.0,3.0")
    ap.add_argument("--merge", action="store_true")
    ap.add_argument("--force", action="store_true", help="rerun even if the record exists")
    args = ap.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    if args.merge:
        merge()
        raise SystemExit

    zs = [float(z) for z in args.zs.split(",")] if INIT == "asym" else [None]
    print(f"INIT={INIT}  MASK={MASK}  N={N}  n_steps={N_STEPS}  burn_in={BURN_IN}  ridge={RIDGE}  out={OUT}")
    for seed in parse_seeds(args.seeds):
        for z in zs:
            p = rec_path(seed, z)
            if p.exists() and not args.force:
                print(f"skip seed {seed} z={z} (record exists)")
                continue
            res, rec = run_one(seed, z)
            np.save(OUT / f"params_seed{seed}_z{'seed' if z is None else f'{z:.1f}'}.npy", np.asarray(res.params))
            json.dump(rec, open(p, "w"), default=float)
            print(f"seed {seed:2d} z={z}: emerged={rec['emerged']} soft_mse={rec['soft_mse']:.4f} "
                  f"norm={rec['norm_mse']:.3f} new_edges={rec['new_edges']} acyclic={rec['acyclic']}", flush=True)

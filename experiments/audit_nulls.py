"""
audit_nulls.py

Three checks raised by reviewers, all through the same hardening -> simulate -> tie-aware cycle test.

A. Cycle specificity (no new simulation of random graphs): for the 107 discovered oscillators with canonical weights,
   how many distinct directed cycles through ALL active neurons does the graph admit (the test already requires the
   observed order to be a cycle through every active neuron), and the pass rate among networks where that cycle is
   unique (a pass there means the one cycle the graph allows, i.e. the specific cycle).
B. Heterogeneous legal weights: each edge gets its own eps_e, delta_e with 0 < eps_e < delta_e/(delta_e+1)
   (eps_e ~ U(0.02, min(0.25, 0.95*delta_e/(delta_e+1))), delta_e ~ U(0.25, 1.0)), 5 draws per circuit.
   Reports outcome split, cycle pass vs graph-specific chance, and per-circuit persistence of outcome class and
   cycle status relative to the canonical (0.1, 0.5) hardening.
C. Random-graph nulls through the same pipeline (N_RAND graphs each, env N_RAND, default 300):
   (i) density-matched random digraphs (edge count drawn from the discovered circuits), and
   (ii) the same, conditioned on containing a Hamiltonian cycle (matching the 92% of discovered graphs).
   Theta = median trained theta (dynamics are homogeneous in theta).

Standalone / non-invasive. Run from the folder with results/, next to the other scripts. Env: SEED (default 0).
"""
import os
from itertools import permutations
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from check_order_ties import compatible, group_phases, null_fraction_groups
from check_order_vs_hamiltonian import edge_matrix, is_cycle
from compare_order_cutoffs import BURN_IN, RESCUE_SCALE, RESCUE_TRIES, WINDOW, analyze, dt, is_flat
from oscillon.ctln.dynamics import simulate
from support_sequence_analysis import spec, x0

RESULTS_DIR = Path("results")
FRAC, TOL = 0.05, 0.02
N_RAND = int(os.environ.get("N_RAND", 300))
SEED = int(os.environ.get("SEED", 0))
N = spec.n_total
ROWS, COLS, _, _, W_CROSS = spec.compile_indices()
ROWS, COLS = jnp.asarray(ROWS), jnp.asarray(COLS)
N_EDGES = spec.n_edges


def build_W(present, eps, delta):
    w_off = np.where(present, -1.0 + eps, -1.0 - delta)
    return jnp.asarray(W_CROSS).at[ROWS, COLS].set(jnp.asarray(w_off))


def run(W, th, start):
    return np.asarray(simulate(W, jnp.full((N,), th), start, dt=dt, n_steps=BURN_IN + WINDOW))[BURN_IN:]


def evaluate(W, th, rng):
    """-> (kind, hit, expected, n_cycles_through_active) ; hit/expected None unless >=3 active neurons."""
    xs = run(W, th, x0)
    if is_flat(xs):
        for _ in range(RESCUE_TRIES):
            xt = run(W, th, x0 + RESCUE_SCALE * jnp.asarray(rng.standard_normal(np.shape(x0))))
            if not is_flat(xt):
                xs = xt
                break
        else:
            return "fixed point", None, None, None
    a = analyze(xs, FRAC)
    if a is None:
        return "other", None, None, None
    _, phases = a
    k = len(phases)
    kind = "five-node" if k == 5 else "subset"
    if k < 3:
        return kind, None, None, None
    E = edge_matrix(np.asarray(W))
    active = sorted(phases)
    groups = group_phases(phases, TOL)
    hit = compatible(groups, E)
    exp = null_fraction_groups(active, [len(g) for g in groups], E)
    ncyc = sum(is_cycle([active[0]] + list(p), E) for p in permutations(active[1:]))
    return kind, hit, exp, ncyc


def has_ham_cycle(present):
    E = edge_matrix(np.asarray(build_W(present, 0.1, 0.5)))
    return any(is_cycle([0] + list(p), E) for p in permutations(range(1, N)))


def summarize(tag, rows):
    kinds = {"five-node": 0, "fixed point": 0, "subset": 0, "other": 0}
    n = hit = 0
    exp = 0.0
    for kind, h, e, _ in rows:
        kinds[kind] += 1
        if h is not None:
            n += 1
            hit += h
            exp += e
    print(f"{tag}: five-node/fixed/subset {kinds['five-node']}/{kinds['fixed point']}/{kinds['subset']}"
          f" (other {kinds['other']});  cycle test {hit}/{n}  (chance {exp:.1f})")


if __name__ == "__main__":
    rng = np.random.default_rng(SEED)
    files = sorted(RESULTS_DIR.glob("params_seed*_z*.npy"))
    params = [jnp.asarray(np.load(pf)) for pf in files]
    thetas = np.array([float(p[N_EDGES]) for p in params])
    presents = [np.asarray(jax.nn.sigmoid(p[:N_EDGES])) > 0.5 for p in params]
    counts = [int(p.sum()) for p in presents]
    print("=" * 78)

    # A + canonical baseline
    canon = [evaluate(build_W(pr, 0.1, 0.5), th, rng) for pr, th in zip(presents, thetas)]
    print("A. canonical hardening (eps 0.1, delta 0.5)")
    summarize("   discovered", canon)
    nc = [c[3] for c in canon if c[3] is not None]
    print(f"   distinct cycles through all active neurons, per oscillator (k>=3): "
          f"{ {v: nc.count(v) for v in sorted(set(nc))} }")
    uniq = [c for c in canon if c[3] == 1]
    print(f"   networks whose graph admits exactly one such cycle: {len(uniq)}; "
          f"order followed it in {sum(c[1] for c in uniq)} (chance {sum(c[2] for c in uniq):.1f})")

    # B
    print("-" * 78)
    print("B. heterogeneous legal weights (5 draws per circuit)")
    all_rows, same_kind, same_cyc, tot, testable = [], 0, 0, 0, 0
    for _ in range(5):
        draw = []
        for pr, th, base in zip(presents, thetas, canon):
            d = rng.uniform(0.25, 1.0, N_EDGES)
            e = rng.uniform(0.02, np.minimum(0.25, 0.95 * d / (d + 1)))
            res = evaluate(build_W(pr, e, d), th, rng)
            draw.append(res)
            tot += 1
            same_kind += res[0] == base[0]
            if res[1] is not None and base[1] is not None:
                same_cyc += res[1] == base[1]
                testable += 1
        all_rows += draw
    summarize("   pooled over draws", all_rows)
    print(f"   outcome class same as canonical: {same_kind}/{tot};  cycle status same (where both testable): {same_cyc}/{testable}")

    # C
    print("-" * 78)
    print(f"C. random graphs through the same pipeline ({N_RAND} each; theta = median trained {np.median(thetas):.3f})")
    th0 = float(np.median(thetas))
    er, ham = [], []
    while len(er) < N_RAND:
        m = counts[rng.integers(len(counts))]
        pr = np.zeros(N_EDGES, bool)
        pr[rng.choice(N_EDGES, m, replace=False)] = True
        er.append((pr, evaluate(build_W(pr, 0.1, 0.5), th0, rng)))
    while len(ham) < N_RAND:
        m = counts[rng.integers(len(counts))]
        pr = np.zeros(N_EDGES, bool)
        pr[rng.choice(N_EDGES, m, replace=False)] = True
        if has_ham_cycle(pr):
            ham.append((pr, evaluate(build_W(pr, 0.1, 0.5), th0, rng)))
    summarize("   density-matched random digraphs      ", [r for _, r in er])
    summarize("   ... conditioned on a Hamiltonian cycle", [r for _, r in ham])
    print("=" * 78)

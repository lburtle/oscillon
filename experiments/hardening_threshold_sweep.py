"""
hardening_threshold_sweep.py

Sensitivity of the audit to the hardening threshold. hardened_weights() makes an edge present when
sigmoid(z) > 0.5. Here the threshold tau is swept (default 0.3 .. 0.7). For each tau:
  * number of edges that differ from the tau=0.5 graph (mean per circuit, and circuits with any change),
  * outcome split: five-node oscillation / fixed point / subset oscillation
    (same x0 start; flat-at-x0 cases rescued from small perturbations as in compare_order_cutoffs.py;
     five-node vs subset = number of active neurons at the 5% cutoff, so the tau=0.5 row may differ slightly
     from the audit's 97/43/10 which used measured fixed-point supports; compare the two),
  * directed-cycle test: firing order is a graph cycle of the tau-hardened graph, with the graph-specific chance.

Also prints how many gates sit near the boundary (0.4 < sigmoid(z) < 0.6).

Standalone / non-invasive. Run from the folder with results/, next to compare_order_cutoffs.py and
check_order_vs_hamiltonian.py.
"""
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from check_order_vs_hamiltonian import edge_matrix
from check_order_ties import compatible, group_phases, null_fraction_groups
from compare_order_cutoffs import BURN_IN, RESCUE_SCALE, RESCUE_TRIES, WINDOW, analyze, dt, is_flat
from oscillon.ctln.dynamics import simulate
from support_sequence_analysis import spec, x0

RESULTS_DIR = Path("results")
TAUS = (0.3, 0.4, 0.5, 0.6, 0.7)
FRAC = 0.05
TOL = 0.02   # neurons firing within this fraction of the period are tied


def hardened_weights_tau(params, tau):
    """Copy of hardened_weights() from support_sequence_analysis.py with the gate threshold as a parameter."""
    n = spec.n_total
    rows, cols, eps_e, delta_e, W_cross = spec.compile_indices()
    eps_j, delta_j = jnp.asarray(eps_e), jnp.asarray(delta_e)
    n_edges = spec.n_edges
    if spec.learn_theta:
        theta = params[n_edges:]
        if theta.shape[0] == 1:
            theta = jnp.full((n,), theta[0])
    else:
        theta = jnp.full((n,), spec.theta_init)
    present = jax.nn.sigmoid(params[:n_edges]) > tau
    w_off = jnp.where(present, -1.0 + eps_j, -1.0 - delta_j)
    W = jnp.asarray(W_cross).at[jnp.asarray(rows), jnp.asarray(cols)].set(w_off)
    return W, theta, np.asarray(present)


def run(W, th, x0_in=x0):
    return np.asarray(simulate(W, th, x0_in, dt=dt, n_steps=BURN_IN + WINDOW))[BURN_IN:]


if __name__ == "__main__":
    files = sorted(RESULTS_DIR.glob("params_seed*_z*.npy"))
    allp = [jnp.asarray(np.load(pf)) for pf in files]
    n_edges = spec.n_edges
    gates = np.concatenate([np.asarray(jax.nn.sigmoid(p[:n_edges])) for p in allp])
    near = ((gates > 0.4) & (gates < 0.6)).mean()
    per_circuit = [int((np.abs(np.asarray(jax.nn.sigmoid(p[:n_edges])) - 0.5) < 0.1).sum()) for p in allp]
    print("=" * 74)
    print(f"{len(files)} circuits, {n_edges} gates each; gates within 0.1 of 0.5: {near:.1%} overall; "
          f"circuits with >=1 such gate: {sum(c > 0 for c in per_circuit)}; max per circuit: {max(per_circuit)}")
    base_present = [hardened_weights_tau(p, 0.5)[2] for p in allp]

    for tau in TAUS:
        rng = np.random.default_rng(0)
        counts = {"five-node": 0, "fixed point": 0, "subset": 0}
        rescued = n = hit = 0
        exp = 0.0
        changed, changed_circuits = 0, 0
        for p, bp in zip(allp, base_present):
            W, th, present = hardened_weights_tau(p, tau)
            diff = int((present != bp).sum())
            changed += diff
            changed_circuits += diff > 0
            xs = run(W, th)
            if is_flat(xs):
                ok = False
                for _ in range(RESCUE_TRIES):
                    xs_try = run(W, th, x0 + RESCUE_SCALE * jnp.asarray(rng.standard_normal(np.shape(x0))))
                    if not is_flat(xs_try):
                        xs, ok = xs_try, True
                        break
                if not ok:
                    counts["fixed point"] += 1
                    continue
                rescued += 1
            a = analyze(xs, FRAC)
            if a is None:
                continue
            _, phases = a
            k = len(phases)
            counts["five-node" if k == 5 else "subset"] += 1
            if k >= 3:
                E = edge_matrix(np.asarray(W))
                active = sorted(phases)
                groups = group_phases(phases, TOL)
                hit += compatible(groups, E)
                exp += null_fraction_groups(active, [len(g) for g in groups], E)
                n += 1
        tot = sum(counts.values())
        print("-" * 74)
        print(f"tau={tau}: edges changed vs tau=0.5: {changed} total in {changed_circuits} circuits "
              f"(mean {changed / len(allp):.2f} per circuit)")
        print(f"   outcome split {counts['five-node']}/{counts['fixed point']}/{counts['subset']} "
              f"(five-node / fixed point / subset; n={tot}; rescued from frozen-at-x0: {rescued})")
        print(f"   firing order is a graph cycle, ties within 2% of the period allowed (forward): {hit}/{n}   (expected by chance {exp:.1f})")
    print("=" * 74)

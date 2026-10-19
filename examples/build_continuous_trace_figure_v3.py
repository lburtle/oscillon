"""
build_continuous_trace_figure_v3.py

Rebuilds the three-panel figure with example selection that matches what each panel claims.
Differences from v2:

  * Oscillation example: v2 took the first non-frozen five-node case, which can show co-firing pairs in
    the hardened row. v3 keeps only oscillators where (a) all five neurons are active in both the trained
    and hardened runs, (b) the cyclic firing order is preserved (the audit's own test: analyze +
    cyclic_order), (c) the hardened phases are well separated (no co-firing), and (d) the run was not
    frozen at x0. Among those it picks the one whose hardened/trained period ratio is closest to the
    sweep median (TARGET_RATIO = 1.19), so the panel shows "order kept, time rescaled".
  * Subset-oscillation example: titled "Subset oscillation", not "collapse". The pick is controlled by
    SUBSET_PICK: "most_visible" (the case where the dropped neuron was most active in the trained network)
    or "median" (the representative case). The full table of the subset oscillations is printed so the
    caption can say how typical the choice is.
  * Fixed-point example: unchanged (cleanest flat case), plus a printout of the numbers the caption quotes.

Standalone / non-invasive: reads existing params, writes continuous_trace_figure_v3.{pdf,png} and prints a
selection report. Needs the same helper modules as v2 plus analyze_hardened_oscillators.py and
compare_period_and_order.py in the same folder. Run from the folder that contains results/.
"""
from pathlib import Path

import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

from analyze_hardened_oscillators import hard_oscillation, sim
from compare_period_and_order import analyze, cyclic_order
from diagnose_boundedness import hard_len1_is_genuine_fixedpoint
from graph_rules_check import measured_fixedpoint_support
from oscillon.dynamics import simulate
from support_sequence_analysis import hardened_weights, p2m, parse_seed_zstd, spec, x0

BURN_IN = 200
REF_PERIOD = 2.0 * np.pi
dt = 0.1
N_STEPS = int(BURN_IN + 8 * (REF_PERIOD / dt))
RESULTS_DIR = Path("results")

STUCK_SEEDS = {(10, 1.0), (12, 3.0), (15, 3.0), (19, 2.0), (24, 1.0), (25, 1.0), (25, 2.0), (27, 3.0),
               (2, 1.0), (30, 1.0), (34, 2.0), (37, 1.0), (37, 3.0), (38, 3.0), (47, 3.0), (49, 3.0)}

TARGET_RATIO = 1.19       # median hardened/trained period ratio over the 107 oscillators
RATIO_BAND = 0.15         # allowed |hardened/trained period ratio - TARGET_RATIO| for the oscillation example
ACTIVE_STD_FRAC = 0.05    # same "active neuron" cutoff as the audit
SUBSET_PICK = "most_visible"   # or "median"
OUT = "continuous_trace_figure_v3"


def hard_trajectory(params):
    W, th = hardened_weights(spec, params)
    return np.asarray(simulate(W, th, x0, dt=dt, n_steps=N_STEPS))[BURN_IN:]


def soft_trajectory(params):
    W, th = p2m(params)
    return np.asarray(simulate(W, th, x0, dt=dt, n_steps=N_STEPS))[BURN_IN:]


def peak_times(x):
    hi = 0.5 * x.max()
    return np.where((x[1:-1] > x[:-2]) & (x[1:-1] > x[2:]) & (x[1:-1] > hi))[0] + 1


def phase_fractions(xs):
    """Firing phase of each active neuron as a fraction of the reference neuron's peak spacing."""
    std = xs.std(axis=0)
    active = [i for i in range(xs.shape[1]) if std[i] >= ACTIVE_STD_FRAC * std.max()]
    ref = max(active, key=lambda i: std[i])
    pk = peak_times(xs[:, ref])
    if len(pk) < 2:
        return None
    period = float(np.mean(np.diff(pk)))
    phases = {}
    for i in active:
        pi = peak_times(xs[:, i])
        if len(pi) == 0:
            return None
        phases[i] = float(((pi[0] - pk[0]) / period) % 1.0)
    return phases


def min_gap(phases):
    v = np.sort(np.array(list(phases.values())))
    return float(np.diff(np.concatenate([v, [v[0] + 1.0]])).min())


if __name__ == "__main__":
    osc, flat_fp, subset, slow_fp = [], [], [], []
    REJECT = {}

    for pf in sorted(RESULTS_DIR.glob("params_seed*_z*.npy")):
        seed, z = parse_seed_zstd(pf)
        params = jnp.asarray(np.load(pf))
        support = measured_fixedpoint_support(pf)

        if len(support) == 5:
            reject = REJECT.setdefault
            if (seed, z) in STUCK_SEEDS:
                reject("stuck", []).append((seed, z)); continue
            try:
                res = hard_oscillation(params)
                if res is None or res[1]:          # fixed point, or frozen at x0
                    reject("hard_oscillation None/frozen", []).append((seed, z)); continue
                xs_h = res[0]
                xs_s = sim(params, "soft", x0)
                a_s, a_h = analyze(xs_s), analyze(xs_h)
                if a_s is None or a_h is None:
                    reject("analyze None", []).append((seed, z)); continue
                (per_s, ph_s), (per_h, ph_h) = a_s, a_h
                shared = set(ph_s) & set(ph_h)
                if len(shared) != 5:
                    reject(f"shared active != 5", []).append((seed, z, len(shared))); continue
                if cyclic_order(ph_s, shared) != cyclic_order(ph_h, shared):
                    reject("order changed", []).append((seed, z)); continue
                f_s, f_h = phase_fractions(xs_s), phase_fractions(xs_h)
                if f_s is None or f_h is None or len(f_s) != 5 or len(f_h) != 5:
                    reject("phase_fractions failed", []).append((seed, z)); continue
                gap = min(min_gap(f_s), min_gap(f_h))
                ratio = per_h / per_s
                sd_h = np.asarray(xs_h).std(axis=0)
                amp = float(sd_h.min() / sd_h.max())   # weakest hardened neuron relative to strongest
                osc.append((abs(ratio - TARGET_RATIO), -gap, seed, z, ratio, gap, params, amp))
            except Exception as e:   # signature mismatch etc.: report the first one in full
                reject(f"EXCEPTION {type(e).__name__}: {e}", []).append((seed, z))
                if sum(len(v) for k, v in REJECT.items() if k.startswith("EXCEPTION")) == 1:
                    import traceback; traceback.print_exc()
            continue

        max_rel = max(hard_len1_is_genuine_fixedpoint(pf)["tail_range_rel_to_scale"])
        if max_rel < 1e-3:
            flat_fp.append((max_rel, seed, z, params))
        else:
            # keep only real oscillations: exclude slowly converging fixed points (amplitude shrinking)
            xs = hard_trajectory(params)
            q = len(xs) // 4
            a_first = np.ptp(xs[:q], axis=0).max()
            a_last = np.ptp(xs[-q:], axis=0).max()
            if a_last >= 0.8 * a_first:
                subset.append((max_rel, seed, z, params))
            else:
                slow_fp.append((seed, z, a_last / max(a_first, 1e-12)))

    print(f"five-node cases kept: {len(osc)}; rejections by stage:")
    for k, v in REJECT.items():
        print(f"   {k}: {len(v)}   e.g. {v[:4]}")
    print(f"fixed-point (flat <1e-3): {len(flat_fp)}; slow-converging excluded from subset: {len(slow_fp)}; "
          f"subset oscillation candidates: {len(subset)}")
    if not osc or not flat_fp or not subset:
        raise RuntimeError(f"missing candidates: osc={len(osc)} fixed_point={len(flat_fp)} subset={len(subset)}")

    # ---- oscillation example -------------------------------------------------------------------------
    # Order-preserved, all-five-active cases. Rank by the smallest gap between neighbouring firing phases
    # (larger = no near-co-firing), restricted to period ratios near the sweep median when possible.
    gaps = np.array([t[5] for t in osc])
    print(f"order-preserved five-node oscillators: {len(osc)}; min-gap quantiles (0/25/50/75/100%): "
          f"{np.round(np.quantile(gaps, [0, .25, .5, .75, 1]), 3).tolist()}")
    print(f"cases with near-co-firing (min phase gap < 0.02 of the period, trained or hardened): "
          f"{int((gaps < 0.02).sum())} of {len(osc)}")
    print(f"distinct (period ratio, gap, amplitude) signatures among {len(osc)}: "
          f"{len({(round(t[4], 2), round(t[5], 3), round(t[7], 2)) for t in osc})}")
    amps = np.array([t[7] for t in osc])
    print(f"weakest-hardened-neuron amplitude (std / max std) quantiles (0/25/50/75/100%): "
          f"{np.round(np.quantile(amps, [0, .25, .5, .75, 1]), 3).tolist()}")
    near = [t for t in osc if abs(t[4] - TARGET_RATIO) <= RATIO_BAND] or osc
    near.sort(key=lambda t: (-t[7], -t[5]))      # all five neurons visible first, then phase separation
    print(f"top 8 within ratio band +/-{RATIO_BAND} of {TARGET_RATIO} "
          f"(seed, z, period ratio, min phase gap, weakest-neuron amplitude):")
    for _, _, s, zz, r, g, _, a in near[:8]:
        print(f"   seed {s} z={zz}  ratio={r:.2f}  min phase gap={g:.3f}  amp={a:.2f}")
    _, _, o_seed, o_z, _, _, o_params, _ = near[0]

    # ---- fixed-point example (cleanest flat case) ----------------------------------------------------
    flat_fp.sort(key=lambda t: t[0])
    _, f_seed, f_z, f_params = flat_fp[0]

    # ---- subset-oscillation example --------------------------------------------------------------------
    rows = []
    for max_rel, s, zz, params in subset:
        ss, sh = soft_trajectory(params).std(axis=0), hard_trajectory(params).std(axis=0)
        dropped = [i for i in range(len(sh)) if sh[i] < ACTIVE_STD_FRAC * sh.max()]
        if not dropped:
            continue
        vis = max(ss[i] / ss.max() for i in dropped)
        rows.append((vis, s, zz, dropped, params))
    if not rows:
        raise RuntimeError("no subset oscillation with a dropped neuron")
    rows.sort(key=lambda t: -t[0])
    print(f"\nsubset oscillations ({len(rows)}): trained activity of the dropped neuron, as a fraction of the "
          f"trained network's most active neuron:")
    for vis, s, zz, dr, _ in rows:
        print(f"   seed {s} z={zz}  dropped={dr}  visibility={vis:.2f}")
    pick = rows[0] if SUBSET_PICK == "most_visible" else rows[len(rows) // 2]
    _, s_seed, s_z, s_dropped, s_params = pick

    examples = [
        ("Five-node oscillation", o_seed, o_z, o_params),
        ("Fixed point", f_seed, f_z, f_params),
        ("Subset oscillation", s_seed, s_z, s_params),
    ]
    print(f"\nSelected: oscillation seed {o_seed} z={o_z}; fixed point seed {f_seed} z={f_z}; "
          f"subset oscillation seed {s_seed} z={s_z} (dropped neuron(s) {s_dropped}, pick={SUBSET_PICK})")

    # ---- numbers the fixed-point caption quotes --------------------------------------------------------
    ts = soft_trajectory(f_params)
    th_ = hard_trajectory(f_params)
    tail = slice(-len(ts) // 2, None)
    print("\ncaption check, fixed-point example:")
    print("   trained tail mean per neuron :", np.round(ts[tail].mean(axis=0), 3).tolist())
    print("   hardened tail mean per neuron:", np.round(th_[tail].mean(axis=0), 3).tolist())

    # ---- figure ------------------------------------------------------------------------------------
    fig, axes = plt.subplots(2, 3, figsize=(9, 3.6), sharex=True)
    for col, (title, seed, z, params) in enumerate(examples):
        for row, (xs, label) in enumerate([(soft_trajectory(params), "Trained"),
                                           (hard_trajectory(params), "Hardened")]):
            mat = np.clip(xs.T, 0, None)
            vmax = mat.max() if mat.max() > 0 else 1.0
            ax = axes[row, col]
            ax.imshow(mat, aspect="auto", cmap="Greys", vmin=0, vmax=vmax, interpolation="nearest")
            ax.set_yticks(range(mat.shape[0]))
            ax.set_yticklabels([str(i) for i in range(mat.shape[0])], fontsize=7)
            if col == 0:
                ax.set_ylabel(f"{label}\nneuron", fontsize=8)
            if row == 0:
                ax.set_title(f"{title}\n(seed {seed}, z={z})", fontsize=8)
            if row == 1:
                ax.set_xlabel("time step", fontsize=8)
            ax.tick_params(labelsize=7)
    fig.tight_layout()
    fig.savefig(f"{OUT}.pdf", bbox_inches="tight")
    fig.savefig(f"{OUT}.png", bbox_inches="tight", dpi=200)
    print(f"\nwrote {OUT}.pdf and {OUT}.png")

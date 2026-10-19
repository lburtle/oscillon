"""
readout_refit_period_matched.py

Is the readout failure after hardening only a period mismatch? For each hardened oscillator (same 107: rescued from
frozen-at-x0 as in the other scripts) the ridge readout is refit on the hardened trajectory against

  original      : sinusoid targets at the reference period 2*pi (the training targets), window = 200 burn-in +
                  8 reference periods (the protocol of the existing refit; should reproduce median MSE ~0.99),
  period-matched: the same five sinusoids (same amplitudes and phase spread) at the HARDENED network's own measured
                  period, window = 8 of its periods taken after a long burn-in.

Also reported: the constant-mean baseline (mean target variance, ~1.05) and the trained (soft) network refit under
both target periods as a control. MSE is raw (mean over time and channels), as in the existing numbers.

fit_readout and make_targets are copied from the user's code (final fit_readout, numpy float64) so the script
is self-contained. Ridge values: env RIDGES (default "1e-6,1e-4,1e-2"). Original-protocol window: env BURN (200) and NSTEPS (default 702; try 2000).
Standalone / non-invasive. Run from the folder with results/, next to the other scripts.
"""
import os
from pathlib import Path

import jax.numpy as jnp
import numpy as np

from compare_order_cutoffs import RESCUE_SCALE, RESCUE_TRIES, analyze, dt, is_flat
from oscillon.dynamics import simulate
from support_sequence_analysis import hardened_weights, p2m, spec, x0

RESULTS_DIR = Path("results")
RIDGES = [float(r) for r in os.environ.get("RIDGES", "1e-6,1e-4,1e-2").split(",")]
REF_PERIOD = 2.0 * np.pi
BURN_ORIG = int(os.environ.get("BURN", 200))
N_ORIG = int(os.environ.get("NSTEPS", int(200 + 8 * (REF_PERIOD / dt))))   # default 702; the n_cycle script used 2000
BURN_LONG, N_LONG = 1000, 4000
N = spec.n_total


def fit_mse(xs, targets, ridge):
    xs, targets = np.asarray(xs, float), np.asarray(targets, float)
    x_mean, y_mean = xs.mean(0), targets.mean(0)
    Xc, Yc = xs - x_mean, targets - y_mean
    G = Xc.T @ Xc
    ridge_eff = ridge * max(np.mean(np.diag(G)), 1.0)
    theta = np.linalg.solve(G + ridge_eff * np.eye(xs.shape[1]), Xc.T @ Yc)
    R = theta.T
    b = y_mean - R @ x_mean
    return float(np.mean((xs @ R.T + b - targets) ** 2))


def make_targets(n, t, period):
    omega = 2.0 * np.pi / period
    return np.stack([(1.0 + 0.5 * (k % 3)) * np.sin(omega * t + 2.0 * np.pi * k / n) for k in range(n)], axis=1)


def period_est(xs):
    """Peak-to-peak period (time units) of the highest-variance neuron; None if no clear cycle."""
    x = np.asarray(xs[:, int(np.argmax(np.asarray(xs).std(0)))])
    hi = 0.5 * x.max()
    pk = np.where((x[1:-1] > x[:-2]) & (x[1:-1] > x[2:]) & (x[1:-1] > hi))[0]
    return None if len(pk) < 2 else float(np.mean(np.diff(pk))) * dt


def sim(W, th, start, n_steps):
    return np.asarray(simulate(W, th, start, dt=dt, n_steps=n_steps))


def both_protocols(W, th, start, ridge_list):
    """-> dict with original / period-matched MSE per ridge, baseline, measured period (or None)."""
    long_w = sim(W, th, start, N_LONG)[BURN_LONG:]
    period = period_est(long_w)
    xs_o = sim(W, th, start, N_ORIG)[BURN_ORIG:]
    tgt_o = make_targets(N, np.arange(len(xs_o)) * dt, REF_PERIOD)
    out = {"period": period, "base": float(tgt_o.var(0).mean()),
           "orig": {r: fit_mse(xs_o, tgt_o, r) for r in ridge_list}}
    if period is not None and 8 * period / dt <= len(long_w):
        m = int(round(8 * period / dt))
        tgt_p = make_targets(N, np.arange(m) * dt, period)
        out["pm"] = {r: fit_mse(long_w[:m], tgt_p, r) for r in ridge_list}
    return out


def rescaled_mse(W, th, start, period_r, ridge_list):
    """Targets at period_r (= 2pi x hardened/trained fundamental ratio): the same harmonic of the hardened cycle."""
    long_w = sim(W, th, start, N_LONG)[BURN_LONG:]
    m = int(round(8 * period_r / dt))
    if m > len(long_w):
        return None
    tgt = make_targets(N, np.arange(m) * dt, period_r)
    return {r: fit_mse(long_w[:m], tgt, r) for r in ridge_list}


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    rows = []
    for pf in sorted(RESULTS_DIR.glob("params_seed*_z*.npy")):
        params = jnp.asarray(np.load(pf))
        W, th = hardened_weights(spec, params)
        start = x0
        w = sim(W, th, start, N_LONG)[BURN_LONG:]
        if is_flat(w):
            for _ in range(RESCUE_TRIES):
                s2 = x0 + RESCUE_SCALE * jnp.asarray(rng.standard_normal(np.shape(x0)))
                w2 = sim(W, th, s2, N_LONG)[BURN_LONG:]
                if not is_flat(w2):
                    start, w = s2, w2
                    break
            else:
                continue
        a = analyze(w, 0.05)
        if a is None:
            continue
        k = len(a[1])
        res = both_protocols(W, th, start, RIDGES)
        Ws, ths = p2m(params)
        res["trained"] = both_protocols(Ws, ths, x0, RIDGES)
        res["k"] = k
        tp_ = res["trained"]["period"]
        if res["period"] and tp_:
            res["rs"] = rescaled_mse(W, th, start, REF_PERIOD * res["period"] / tp_, RIDGES)
        rows.append(res)

    print("=" * 78)
    print(f"{len(rows)} hardened oscillators; measured hardened period / 2pi: median "
          f"{np.median([r['period'] / REF_PERIOD for r in rows if r['period']]):.2f}")
    hist = {}
    for r in rows:
        tp_ = r["trained"]["period"]
        key = round(tp_ / REF_PERIOD) if tp_ else None
        hist[key] = hist.get(key, 0) + 1
    print(f"trained fundamental period / 2pi (rounded): {dict(sorted(hist.items(), key=lambda kv: str(kv[0])))}")
    for label, sel in (("all oscillators", lambda r: True), ("five-node only", lambda r: r["k"] == 5)):
        sub = [r for r in rows if sel(r) and "pm" in r]
        print("-" * 78)
        print(f"{label}: n = {len(sub)}; constant-mean baseline (median) {np.median([r['base'] for r in sub]):.3f}")
        for ridge in RIDGES:
            o = np.array([r["orig"][ridge] for r in sub])
            p = np.array([r["pm"][ridge] for r in sub])
            to = np.array([r["trained"]["orig"][ridge] for r in sub])
            tp = np.array([r["trained"]["pm"][ridge] for r in sub if "pm" in r["trained"]])
            rs = np.array([r["rs"][ridge] for r in sub if r.get("rs")])
            print(f"  ridge {ridge:g}: time-rescaled 2pi targets (same harmonic): median {np.median(rs):.3f} "
                  f"(< 0.25 in {int((rs < 0.25).sum())}/{len(rs)}, < 0.5 in {int((rs < 0.5).sum())}/{len(rs)})")
            print(f"  ridge {ridge:g}: hardened, 2pi targets (original) median MSE {np.median(o):.3f};  "
                  f"hardened, own-period targets {np.median(p):.3f}  "
                  f"(own-period MSE < 0.25 in {int((p < 0.25).sum())}/{len(p)}, < 0.5 in {int((p < 0.5).sum())}/{len(p)});  "
                  f"trained, 2pi {np.median(to):.3f}, own-period {np.median(tp):.3f}")
    print("=" * 78)

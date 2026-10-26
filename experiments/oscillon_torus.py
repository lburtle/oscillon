"""
gCTLN quasiperiodic torus pipeline (single file).

Idea:
  Two CTLN 3-cycles ("triads") with detuned natural frequencies (via
  different effective time constants), weakly cross-coupled. In the
  quasiperiodic regime (winner-take-all -> antiphase -> QUASIPERIODIC
  -> synchronization), the 6-dim reservoir state traces a 2-torus.
  PCA down to the top 3 principal components reveals that torus as an
  embedded 3D surface -- no target curve is ever fit; the shape is a
  property of the dynamics, the readout is just a linear window onto it.

Pipeline stages:
  1. build_W        - construct the 6x6 competitive weight matrix for
                       two 3-cycles + symmetric cross-inhibition
  2. simulate        - forward-Euler CTLN dynamics, tau per-node
  3. sweep           - scan cross-coupling strength eps_cross, score
                       each run by (freq-ratio rationality) x (torus
                       fatness from PCA eigenvalues)
  4. render          - long run at best eps_cross, PCA -> 3D, multi-
                       stroke white-on-black rendering, save PNG
"""

from fractions import Fraction

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ----------------------------------------------------------------------
# 1. Topology: two 3-cycles + symmetric weak cross-coupling
# ----------------------------------------------------------------------
# NOTE on parameters: a CTLN 3-cycle only has a stable oscillating limit
# cycle (rather than collapsing to the symmetric interior fixed point,
# or to silence) when 0 < delta < eps/(1+eps). Detuning the two triads'
# frequencies is done via *different delta per triad* (each triad is
# its own independently-tuned oscillator), NOT via a per-block time
# constant -- a shared forward-Euler dt applied at two different
# effective step sizes (dt/tau) changes discretization behavior enough
# to knock one triad out of its oscillating basin. Different delta is
# both more robust and more principled: two structurally distinct
# 3-cycles, each independently oscillating at its own natural rate.
def _triad_block(eps, delta):
    """Standard 3x3 CTLN competitive block for a single 3-cycle 0->1->2->0."""
    Wb = np.full((3, 3), -1.0 - eps)
    np.fill_diagonal(Wb, 0.0)
    for j, i in [(0, 1), (1, 2), (2, 0)]:
        Wb[i, j] = -1.0 + delta
    return Wb


def build_W(eps=0.25, delta_A=0.1, delta_B=0.05, eps_cross=0.0):
    """6x6 competitive weight matrix, block-diagonal at eps_cross=0 (two
    fully independent 3-cycles). eps_cross is the *only* source of
    cross-triad coupling -- it symmetrically inhibits every A node from
    every B node and vice versa, going from decoupled (0) up through
    winner-take-all as it grows, matching the bifurcation sequence
    winner-take-all -> antiphase -> quasiperiodic -> synchronization."""
    W = np.zeros((6, 6))
    W[0:3, 0:3] = _triad_block(eps, delta_A)
    W[3:6, 3:6] = _triad_block(eps, delta_B)
    W[0:3, 3:6] = -eps_cross
    W[3:6, 0:3] = -eps_cross
    return W


# ----------------------------------------------------------------------
# 2. Dynamics: dx/dt = -x + relu(Wx + theta)
# ----------------------------------------------------------------------
def simulate(W, theta=1.0, dt=0.02, n_steps=40000, x0=None, seed=0):
    n = W.shape[0]
    if x0 is None:
        # asymmetric init: symmetric/uniform starts can land exactly on
        # the (unstable but numerically sticky) interior fixed point
        rng = np.random.default_rng(seed)
        base = np.array([0.3, 0.1, 0.02])
        x = np.concatenate([base, base]) * rng.uniform(0.8, 1.2, size=n)
    else:
        x = x0.copy()
    traj = np.empty((n_steps, n))
    for t in range(n_steps):
        drive = W @ x + theta
        x = x + dt * (-x + np.maximum(drive, 0.0))
        traj[t] = x
    return traj


# ----------------------------------------------------------------------
# 3. Sweep cross-coupling: score by freq-ratio rationality x fatness
# ----------------------------------------------------------------------
def dominant_freq(signal, dt):
    signal = signal - signal.mean()
    spec = np.abs(np.fft.rfft(signal))
    freqs = np.fft.rfftfreq(len(signal), d=dt)
    spec[0] = 0.0  # kill DC
    return freqs[np.argmax(spec)]


def score_run(traj, dt, burn):
    x = traj[burn:]
    yA = x[:, 0] + x[:, 1] + x[:, 2]
    yB = x[:, 3] + x[:, 4] + x[:, 5]
    fA, fB = dominant_freq(yA, dt), dominant_freq(yB, dt)
    if fA == 0 or fB == 0:
        return -np.inf, None
    ratio = fA / fB
    frac = Fraction(ratio).limit_denominator(6)
    rational_err = abs(ratio - float(frac))

    Xc = x - x.mean(axis=0)
    cov = Xc.T @ Xc / len(Xc)
    eigvals = np.sort(np.linalg.eigvalsh(cov))[::-1]
    fatness = eigvals[1] / (eigvals[0] + 1e-12)  # 2nd/1st PCA eigenvalue

    score = fatness - 4.0 * rational_err
    info = dict(
        fA=fA,
        fB=fB,
        ratio=ratio,
        frac=frac,
        rational_err=rational_err,
        fatness=fatness,
        eigvals=eigvals,
    )
    return score, info


def sweep(
    eps_cross_values, delta_A=0.06, delta_B=0.07, dt=0.02, n_steps=20000, burn=8000
):
    results = []
    for ec in eps_cross_values:
        W = build_W(delta_A=delta_A, delta_B=delta_B, eps_cross=ec)
        traj = simulate(W, dt=dt, n_steps=n_steps, seed=1)
        s, info = score_run(traj, dt, burn)
        results.append((ec, s, info))
    return results


# ----------------------------------------------------------------------
# 4. Render: long run at best eps_cross -> PCA(3) -> torus figure
# ----------------------------------------------------------------------
def render_torus(
    eps_cross,
    delta_A=0.06,
    delta_B=0.07,
    dt=0.02,
    n_steps=200000,
    burn=20000,
    elev=15,
    azim=-25,
    roll=225,
    transparent=True,
    stroke_color="green",
    outpath="oscillon_torus.png",
):
    W = build_W(delta_A=delta_A, delta_B=delta_B, eps_cross=eps_cross)
    traj = simulate(W, dt=dt, n_steps=n_steps, seed=2)
    X = traj[burn:]

    Xc = X - X.mean(axis=0)
    U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
    P = Xc @ Vt[:3].T  # top-3 PCA coordinates, shape (T, 3)
    P = P / np.max(np.linalg.norm(P, axis=1))  # normalize to unit scale

    bg = "none" if transparent else "black"
    fig = plt.figure(figsize=(8, 6.4), facecolor=bg)
    ax = fig.add_subplot(111, projection="3d")
    ax.set_facecolor(bg)
    if transparent:
        # the 3D panes (the box walls) default to an opaque light fill
        # even when the figure/axes facecolor is transparent -- each
        # pane's fill must be zeroed out individually
        for pane in (ax.xaxis.pane, ax.yaxis.pane, ax.zaxis.pane):
            pane.set_alpha(0.0)

    # multi-stroke rendering: several overlapping thin, low-alpha passes
    # with tiny jitter reproduces the "hand-scratched" photographic look
    rng = np.random.default_rng(0)
    n_strokes = 10
    stride = 2
    for k in range(n_strokes):
        jitter = rng.normal(scale=0.003, size=P.shape)
        Pk = P[::stride] + jitter[::stride]
        ax.plot(
            Pk[:, 0], Pk[:, 1], Pk[:, 2], color=stroke_color, linewidth=0.7, alpha=0.06
        )

    ax.set_axis_off()
    # roll spins the image about the camera's own forward/viewing axis --
    # pure rendering convention, carries no dynamical meaning (unlike
    # elev/azim, which do change which physical direction you're viewing
    # the trajectory from).
    ax.view_init(elev=elev, azim=azim, roll=roll)
    ax.set_box_aspect([1, 1, 1])
    ax.set_xlim(-0.7, 0.7)
    ax.set_ylim(-0.7, 0.7)
    ax.set_zlim(-0.7, 0.7)
    plt.tight_layout()
    fig.savefig(outpath, dpi=220, facecolor=bg, transparent=transparent)
    plt.close(fig)
    return outpath, P


if __name__ == "__main__":
    eps_values = np.linspace(0.0, 0.75, 25)
    results = sweep(eps_values)

    print(
        f"{'eps_cross':>10} {'fA':>7} {'fB':>7} {'ratio':>7} {'~frac':>7} {'err':>7} {'fatness':>8} {'score':>8}"
    )
    best = (-np.inf, None, None)
    for ec, s, info in results:
        if info is None:
            continue
        print(
            f"{ec:10.3f} {info['fA']:7.3f} {info['fB']:7.3f} "
            f"{info['ratio']:7.3f} {info['frac']!s:>7} "
            f"{info['rational_err']:7.3f} {info['fatness']:8.3f} {s:8.3f}"
        )
        if s > best[0]:
            best = (s, ec, info)

    best_score, best_ec, best_info = best
    print(
        f"\nbest eps_cross = {best_ec:.3f}  "
        f"(freq ratio {best_info['ratio']:.3f} ~ {best_info['frac']}, "
        f"fatness {best_info['fatness']:.3f}, score {best_score:.3f})"
    )

    outpath, P = render_torus(best_ec)
    print(f"saved: {outpath}")

import itertools

import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

from oscillon.dynamics import simulate


def _field_np(W, theta, X):
    return -X + np.maximum(X @ np.asarray(W).T + np.asarray(theta), 0.0)


def plot_phase_grid(
    W,
    theta,
    *,
    x0s,
    grid_range=(-0.1, 1.5),
    grid_n=5,
    slice_value=0.0,
    dt=0.1,
    n_steps=1500,
    title="phase space (all neuron pairs)",
    save_path=None,
):
    """Grid of 2D coordinate slices, one per neuron pair, sharing a title.
    Each panel: streamplot of the field on (x_i, x_j) with other axes at slice_value,
    plus projected trajectories from x0s. Slice caveat applies for N>3."""
    W = np.asarray(W)
    theta = np.asarray(theta)
    N = W.shape[0]
    pairs = list(itertools.combinations(range(N), 2))
    ncol = min(3, len(pairs))
    nrow = int(np.ceil(len(pairs) / ncol))

    fig, axes = plt.subplots(
        nrow, ncol, figsize=(4.2 * ncol, 4.0 * nrow), squeeze=False
    )

    # pre-simulate trajectories once (shared across panels)
    trajs = [
        np.asarray(
            simulate(
                jnp.asarray(W),
                jnp.asarray(theta),
                jnp.asarray(x0),
                dt=dt,
                n_steps=n_steps,
            )
        )
        for x0 in x0s
    ]

    g = np.linspace(grid_range[0], grid_range[1], grid_n)
    GX, GY = np.meshgrid(g, g)

    for idx, (i, j) in enumerate(pairs):
        ax = axes[idx // ncol][idx % ncol]
        # lift grid to N-D on this slice
        P = np.full((GX.size, N), slice_value, float)
        P[:, i] = GX.ravel()
        P[:, j] = GY.ravel()
        F = _field_np(W, theta, P)
        U, V = F[:, i].reshape(GX.shape), F[:, j].reshape(GX.shape)
        mag = np.hypot(U, V)
        ax.streamplot(
            GX,
            GY,
            U,
            V,
            density=1.1,
            color=mag,
            cmap="viridis",
            linewidth=0.7,
            arrowsize=0.8,
        )

        # switching manifolds for the two plotted neurons
        xs_line = np.linspace(*grid_range, 200)
        for k in (i, j):
            a_i, a_j = W[k, i], W[k, j]
            const = theta[k] + sum(
                W[k, d] * slice_value for d in range(N) if d not in (i, j)
            )
            if abs(a_j) > 1e-9:
                ax.plot(
                    xs_line,
                    -(a_i * xs_line + const) / a_j,
                    c="gray",
                    lw=0.5,
                    ls="--",
                    alpha=0.5,
                )

        for traj in trajs:
            ax.plot(traj[:, i], traj[:, j], lw=1.0, alpha=0.85)
            ax.scatter(traj[0, i], traj[0, j], c="k", s=18, zorder=5)
            ax.scatter(traj[-1, i], traj[-1, j], c="r", s=32, marker="*", zorder=5)

        ax.set_xlim(grid_range)
        ax.set_ylim(grid_range)
        ax.set_xlabel(f"$x_{{{i}}}$")
        ax.set_ylabel(f"$x_{{{j}}}$")

    # hide any unused panels
    for idx in range(len(pairs), nrow * ncol):
        axes[idx // ncol][idx % ncol].axis("off")

    fig.suptitle(title, fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    if save_path:
        fig.savefig(save_path, dpi=130, bbox_inches="tight")
    return fig

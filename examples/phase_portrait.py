import numpy as np
import matplotlib.pyplot as plt
import jax
import jax.numpy as jnp

from oscillon.dynamics import simulate


def _field(W, theta, X):
    """dx/dt = -x + relu(Wx + theta), batched over rows of X. X: (M, N) -> (M, N)."""
    return -X + jax.nn.relu(X @ np.asarray(W).T + np.asarray(theta))


def plot_phase_portrait(
    W, theta, *,
    dims=(0, 1),              # which two neurons to view, OR ('pca', trajectory)
    x0s=None,                 # list of initial conditions to trace, each (N,)
    grid_range=(-0.2, 2.0),
    grid_n=22,
    slice_value=0.0,          # value held on the non-plotted axes (N>2 slice mode)
    dt=0.1, n_steps=3000,
    title="phase portrait",
    save_path=None,
):
    """
    Vector field + trajectories for a gCTLN, projected to 2D.

    dims: (i, j) plots the (x_i, x_j) plane, other axes held at slice_value.
          ('pca', traj) fits a 2D PCA plane to trajectory `traj` (T,N) and
          projects the field onto it (better for genuinely N-D attractors).
    """
    W = np.asarray(W); theta = np.asarray(theta)
    N = W.shape[0]

    pca_mode = isinstance(dims, tuple) and dims[0] == "pca"
    if pca_mode:
        traj = np.asarray(dims[1])
        mean = traj.mean(0)
        # two leading PCs
        U, S, Vt = np.linalg.svd(traj - mean, full_matrices=False)
        basis = Vt[:2]                        # (2, N)
        def to_plane(x): return (x - mean) @ basis.T      # (…,N)->(…,2)
        def from_plane(p): return mean + p @ basis        # (…,2)->(…,N)
        lo, hi = to_plane(traj).min(0), to_plane(traj).max(0)
        gx = np.linspace(lo[0], hi[0], grid_n)
        gy = np.linspace(lo[1], hi[1], grid_n)
    else:
        i, j = dims
        gx = gy = np.linspace(grid_range[0], grid_range[1], grid_n)

    GX, GY = np.meshgrid(gx, gy)
    P = np.stack([GX.ravel(), GY.ravel()], axis=1)         # (M, 2)

    # lift grid points to N-D, compute field, project back to 2D
    if pca_mode:
        Xfull = from_plane(P)
        F = np.asarray(_field(W, theta, Xfull))
        Fp = F @ basis.T                                   # project field to plane
        U2, V2 = Fp[:, 0], Fp[:, 1]
    else:
        Xfull = np.full((P.shape[0], N), slice_value, float)
        Xfull[:, i] = P[:, 0]; Xfull[:, j] = P[:, 1]
        F = np.asarray(_field(W, theta, Xfull))
        U2, V2 = F[:, i], F[:, j]

    fig, ax = plt.subplots(figsize=(6.5, 6))
    mag = np.hypot(U2, V2).reshape(GX.shape)
    ax.streamplot(GX, GY, U2.reshape(GX.shape), V2.reshape(GX.shape),
                  density=1.3, color=mag, cmap="viridis", linewidth=0.8,
                  arrowsize=0.9)

    # switching manifolds: where (Wx+theta)_k = 0 for each neuron k.
    # In slice mode these are lines in the (i,j) plane; draw the plotted axes' own.
    if not pca_mode:
        xs_line = np.linspace(grid_range[0], grid_range[1], 200)
        for k in range(N):
            a_i, a_j = W[k, i], W[k, j]
            # (Wx+theta)_k = a_i*x_i + a_j*x_j + const = 0, const from sliced axes
            const = theta[k] + sum(W[k, d] * slice_value
                                   for d in range(N) if d not in (i, j))
            if abs(a_j) > 1e-9:
                ax.plot(xs_line, -(a_i * xs_line + const) / a_j,
                        c="gray", lw=0.6, ls="--", alpha=0.5)
        ax.set_xlim(grid_range); ax.set_ylim(grid_range)

    # trajectories
    if x0s is not None:
        for x0 in x0s:
            traj = np.asarray(simulate(jnp.asarray(W), jnp.asarray(theta),
                                       jnp.asarray(x0), dt=dt, n_steps=n_steps))
            pts = to_plane(traj) if pca_mode else traj[:, [dims[0], dims[1]]]
            ax.plot(pts[:, 0], pts[:, 1], lw=1.2, alpha=0.9)
            ax.scatter(*pts[0], c="k", s=30, zorder=5)         # start
            ax.scatter(*pts[-1], c="r", s=40, marker="*", zorder=5)  # end

    ax.set_xlabel("PC1" if pca_mode else f"$x_{{{dims[0]}}}$")
    ax.set_ylabel("PC2" if pca_mode else f"$x_{{{dims[1]}}}$")
    ax.set_title(title)
    if save_path:
        plt.savefig(save_path, dpi=140, bbox_inches="tight")
    return fig

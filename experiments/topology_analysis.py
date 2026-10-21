"""Analytical fixed-point and stability analysis for gCTLNs."""
from __future__ import annotations
import numpy as np


def solve_fixed_point(W, theta, active):
    """Closed-form fixed point on a given active set (support).
    On the active set the relu is identity: (I - W_σσ) x_σ = θ_σ.
    active: index array of neurons assumed to have x*_i > 0."""
    W, theta = np.asarray(W), np.asarray(theta)
    x = np.zeros(theta.shape[0])
    if len(active) == 0:
        return x
    Wss = W[np.ix_(active, active)]
    x[active] = np.linalg.solve(np.eye(len(active)) - Wss, theta[active])
    return x


def check_fixed_point(W, theta, x_star, tol=1e-6):
    """Verify x* is a genuine fixed point: dx/dt = -x + relu(Wx+theta) ≈ 0,
    AND the active-set sign conditions are self-consistent."""
    W, theta = np.asarray(W), np.asarray(theta)
    drive = W @ x_star + theta
    dxdt = -x_star + np.maximum(drive, 0.0)
    residual = np.linalg.norm(dxdt)
    # sign consistency: active neurons have x*>0, inactive have drive<=0
    active = x_star > tol
    consistent = bool(np.all(x_star[active] > 0) and np.all(drive[~active] <= tol))
    return residual, consistent


def local_stability(W, active):
    """Eigenvalues of the local Jacobian (-I + W_σσ) on the active region.
    Fixed point is stable iff all Re(eig) < 0, i.e. all Re(eig(W_σσ)) < 1.
    Returns (eigenvalues_of_Wss, is_stable)."""
    W = np.asarray(W)
    if len(active) == 0:
        return np.array([]), True
    Wss = W[np.ix_(active, active)]
    if not np.all(np.isfinite(Wss)):          # diverged run
        return np.array([]), False            # classify as unstable/failed
    eigs = np.linalg.eigvals(Wss)
    is_stable = bool(np.all(eigs.real < 1.0))
    return eigs, is_stable


def analyze_interior_fixed_point(W, theta):
    """Full analysis assuming ALL neurons active (the interior fixed point
    a limit cycle typically surrounds). Returns a dict."""
    W, theta = np.asarray(W), np.asarray(theta)
    n = W.shape[0]
    active = np.arange(n)
    x_star = solve_fixed_point(W, theta, active)
    residual, consistent = check_fixed_point(W, theta, x_star)
    eigs, stable = local_stability(W, active)

    if eigs.size == 0:
        max_real = float("nan")
        stable = True
    else:
        max_real = float(eigs.real.max())
        stable = bool((eigs.real < 0).all())

    return {
        "x_star": x_star,
        "residual": residual,
        "sign_consistent": consistent,
        "eigs_Wss": eigs,
        "max_real_eig": max_real,
        "stable": stable,
    }

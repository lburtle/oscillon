import numpy as np
import jax.numpy as jnp
from jsonloader import load_runs
from oscillon.ctln.dynamics import simulate
from oscillon.ctln.readout import apply_readout, fit_readout
from topo_ablation_sweep import (BURN_IN, DELTA, DT, EPS, N, N_STEPS, REF_PERIOD, RIDGE, 
                                X0, make_targets, measure_period)


RUNS = "results/uniform_pop512.json"
U = "hyperparameters/uniform_pop512_u.npz"
TARGET_SEEDS = (95, 139)

runs = load_runs(RUNS)
u = np.load(U, allow_pickle=True)["u"]
print("RECORD KEYS:", sorted(runs[0].keys()))
print("U SHAPE:", u.shape, "| min/max:", float(np.min(u)), float(np.max(u)))

bad = [(i, r["seed"], r["z_std"], r["amp"]) for i, r in enumerate(runs) if not r["emerged"]]
print("BAD:", bad)

from topo_ablation_sweep import cycle_emerged

U_ROWS = u.reshape(len(runs), N * (N - 1))          # 150 x 20 gate values, same order as the records (to verify)
OFF = ~np.eye(N, dtype=bool)

def W_from_gates(g20):
    G = np.zeros((N, N)); G[OFF] = g20               # row-major over off-diagonal entries (to verify)
    W = -1 - DELTA + (EPS + DELTA) * G
    np.fill_diagonal(W, 0)
    return W

def soft_mse_amp(W, th):
    thv = jnp.full((N,), th)
    t = jnp.arange(N_STEPS) * DT
    tgt = make_targets(N, t, REF_PERIOD)[BURN_IN:]
    xs = jnp.asarray(simulate(jnp.asarray(W), thv, X0, dt=DT, n_steps=N_STEPS))[BURN_IN:]
    m = min(xs.shape[0], tgt.shape[0]); xs, tgt = xs[:m], tgt[:m]
    R, b = fit_readout(xs, tgt, ridge=RIDGE)
    mse = float(jnp.mean((apply_readout(xs, R, b) - tgt) ** 2))
    _, amp, _ = cycle_emerged(jnp.asarray(W), thv, X0, DT, N_STEPS, BURN_IN)
    return mse, amp

def recover_theta(W, rec):
    grid = np.geomspace(0.05, 30.0, 200)
    err = [abs(soft_mse_amp(W, th)[0] - rec["soft_mse"]) for th in grid]
    th0 = grid[int(np.argmin(err))]
    fine = np.linspace(th0 * 0.93, th0 * 1.07, 141)
    errs = [abs(soft_mse_amp(W, th)[0] - rec["soft_mse"]) for th in fine]
    j = int(np.argmin(errs))
    return fine[j], errs[j]

def diagnose(W, theta, n_long=20000):
    xs = np.asarray(simulate(jnp.asarray(W), theta, X0, dt=DT, n_steps=n_long))
    win = lambda a, b: xs[int(a * n_long): int(b * n_long)]
    best = int(win(0.5, 1.0).std(0).argmax())                       # most active neuron
    return dict(
        std_per_neuron_late=np.round(win(0.8, 1.0).std(0), 3).tolist(),
        std_early=float(win(0.1, 0.2).std(0).max()),
        std_late=float(win(0.8, 1.0).std(0).max()),
        period_neuron0=measure_period(win(0.5, 1.0), DT),
        period_best_neuron=measure_period(np.roll(win(0.5, 1.0), -best, axis=1), DT),   # best neuron -> column 0
        final_support=np.where(xs[-1] > 1e-3)[0].tolist(),
        x_final=np.round(xs[-1], 3).tolist(),
    )

for i, r_seed, z, _ in bad:
    rec, row = runs[i], U_ROWS[i]
    # alignment check that needs no theta: the stored gate statistic must match the gates in this row
    print(f"\n=== record index {i}, seed {r_seed}, z={z} ===")
    print("mean |g-0.5| from u:", round(float(np.mean(np.abs(row - 0.5))), 5), "| stored mean_dist_mid:", round(rec["mean_dist_mid"], 5))
    W = W_from_gates(row)
    th, e = recover_theta(W, rec)
    mse, amp = soft_mse_amp(W, th)
    print(f"recovered theta={th:.4f}; soft_mse={mse:.6f} (stored {rec['soft_mse']:.6f}, err {e:.1e}); amp={amp:.4f} (stored {rec['amp']:.4f})")
    if e < 1e-5 and abs(amp - rec["amp"]) < 1e-3:
        print(diagnose(W, jnp.full((N,), th)))
    else:
        print("could not reproduce the record: do not trust this reconstruction (rerun the seed with W and theta saved)")

th = jnp.full((N,), 1.9578); W = jnp.asarray(W_from_gates(U_ROWS[95]))
for n_steps in (702, 1000, 1500, 2000, 2800):
    em, amp, per = cycle_emerged(W, th, X0, DT, n_steps, BURN_IN)
    print(n_steps, "emerged:", em, "amp:", round(amp, 4), "period:", per)
print("stored amp:", runs[95]["amp"], "stored period:", runs[95]["period"])

print("MULTI START\n")
def multi_start(W, theta, n_starts=20, n_long=20000, seed=0):
    rng = np.random.default_rng(seed)
    out = []
    for s in range(n_starts):
        x0 = jnp.asarray(rng.uniform(0.0, 1.0, size=N))
        xs = np.asarray(simulate(jnp.asarray(W), theta, x0, dt=DT, n_steps=n_long))
        late = xs[int(0.8 * n_long):]
        out.append((float(late.std(0).max()), np.round(xs[-1], 2).tolist()))
    return out

res = multi_start(W, jnp.full((N,), th))
osc = sum(a > 0.05 for a, _ in res)
print(f"{osc}/{len(res)} starts keep oscillating (late std > 0.05)")
for a, xf in res[:6]:
    print(round(a, 3), xf)


def mse_amp_curve(W, ths):
    return [soft_mse_amp(W, th) for th in ths]

def roots_for(W, rec, lo=0.05, hi=30.0, n=600):
    ths = np.geomspace(lo, hi, n)
    vals = mse_amp_curve(W, ths)
    f = np.array([m - rec["soft_mse"] for m, _ in vals])
    out = []
    for k in np.where(np.sign(f[:-1]) != np.sign(f[1:]))[0]:
        a, b = ths[k], ths[k + 1]
        for _ in range(40):                                  # bisection on theta
            mid = 0.5 * (a + b)
            if (soft_mse_amp(W, mid)[0] - rec["soft_mse"]) * (soft_mse_amp(W, a)[0] - rec["soft_mse"]) <= 0:
                b = mid
            else:
                a = mid
        th = 0.5 * (a + b)
        out.append((th, *soft_mse_amp(W, th)))
    return out

for i in (95, 139):
    W = W_from_gates(U_ROWS[i]); rec = runs[i]
    print(f"\nrecord {i}: stored soft_mse {rec['soft_mse']:.6f}, stored amp {rec['amp']:.4f}")
    for th, mse, amp in roots_for(W, rec):
        print(f"  theta={th:.4f}  mse={mse:.6f}  amp={amp:.4f}")


for i, th_ref, amp_ref in [(95, 1.9558, 0.2402), (139, 3.4130, 0.9527)]:
    rec = runs[i]
    th = th_ref * rec["amp"] / amp_ref                     # amp is proportional to theta
    W = W_from_gates(U_ROWS[i])
    mse, amp = soft_mse_amp(W, th)
    print(i, f"theta={th:.4f}  mse={mse:.6f} (stored {rec['soft_mse']:.6f})  amp={amp:.4f} (stored {rec['amp']:.4f})")

th = 4.5705
W = W_from_gates(U_ROWS[95])
print(diagnose(W, jnp.full((N,), th)))

res = multi_start(W, jnp.full((N,), th))
print(sum(a > 0.05 for a, _ in res), "/", len(res), "starts keep oscillating")

print(runs[95]["hamiltonian_cycle"])
print(runs[139]["hamiltonian_cycle"])
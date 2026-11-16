import numpy as np, collections
from jsonloader import load_runs

runs = load_runs("results/uniform_pop512.json")
print(sorted(runs[0].keys()))

# field names
F = dict(emerged="emerged", period="period", mse="soft_mse", z="z_std")
TARGET = 2 * np.pi
BASE = 1.05     # constant mean baseline

rows = []
for r in runs:
    if not r.get(F["emerged"]) or r.get(F["period"]) is None: continue
    k_real = r[F["period"]] / TARGET
    k = int(round(k_real))
    rows.append((r[F["z"]], k, k_real - k, r[F["mse"]] / BASE))
rows = np.array(rows)
print(len(rows), "emerged runs with a measured period")

# MSE by int multiple k (pooled and per z)
for zv in [None, 1.0, 2.0, 3.0]:
    sub = rows if zv is None else rows[rows[:, 0] == zv]
    print("\nz =", zv or "all")
    for k in sorted(set(sub[:, 1])):
        v = sub[sub[:, 1] == k][:, 3]
        q = np.percentile(v, [25, 50, 75])
        print(f"  k={int(k)}: n={len(v):3d}  median={q[1]:.4f}  IQR=[{q[0]:.4f}, {q[2]:.4f}]")

# Does distance from an exact integer multiple matter?
from scipy.stats import spearmanr, mannwhitneyu
print("\nSpearman(|k_real - k|, normalied MSE):", spearmanr(np.abs(rows[:, 2]), rows[:, 3]))

# k=2 vs k=3 test
a, b = rows[rows[:, 1] == 2][:, 3], rows[rows[:, 1] == 3][:, 3]
if len(a) and len(b): print("Mann-Whitney k=2 vs k=3:", mannwhitneyu(a, b))

# within-k=2
k2 = rows[rows[:, 1] == 2]
res = spearmanr(np.abs(k2[:, 2]), k2[:, 3])
print(f"\nwithin k=2 (n={len(k2)}): rho={res.statistic:.3f}, p={res.pvalue:.3g}")

# within-k=3
k3 = rows[rows[:, 1] == 3]
res3 = spearmanr(np.abs(k3[:, 2]), k3[:, 3])
print(f"within k=3 (n={len(k3)}): rho={res3.statistic:.3f}, p={res3.pvalue:.3g}")

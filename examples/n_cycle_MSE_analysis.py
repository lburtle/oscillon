import json, numpy as np
rows = [json.loads(l) for l in open("results/mse_runs.jsonl")]
rows = [r for r in rows if r.get("init") == "asymmetric"]
for label in (True, False):
    g = [r["mse"] for r in rows if r["emerged"] is label]
    if g:
        print(f"emerged={label}  n={len(g)}  median={np.median(g):.4f}  "
              f"IQR=[{np.percentile(g,25):.4f}, {np.percentile(g,75):.4f}]  max={max(g):.4f}")
print(f"discovery rate: {sum(r['emerged'] for r in rows)}/{len(rows)}")

for r in sorted(rows, key=lambda r: -r["mse"])[:5]:
    print(f"{r['seed']:3d} mse={r['mse']:.4f} emerged={r['emerged']} "
              f"per={r['trained_period']} tgt={r['target_period']}")

tail = [18, 8, 46, 38, 19]
core = [r["seed"] for r in sorted(rows, key=lambda r: r["mse"])[:5]]
# for each, count nodes in the recovered cycle from A_from_W

print(core)
import numpy as np
pers = np.array([r["trained_period"] for r in rows if r["trained_period"]])
mses = np.array([r["mse"] for r in rows if r["trained_period"]])
print(np.corrcoef(pers, mses)[0,1])
# and the shape of the period distribution
print(np.round(np.sort(pers), 1))

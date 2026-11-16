import json, numpy as np
from collections import defaultdict
ref = {(r["seed"], r["z_std"]): r for r in json.load(open("results/uniform_pop512.json"))}
rows = [json.loads(l) for l in open("hyperparameters/ablation_soft.jsonl") if l.strip()]
copies = defaultdict(list)                       # occurrence order = file order
seen = defaultdict(int)
for r in rows:
    k = (r["seed"], r["z_std"])
    copies[seen[k]].append(r)
    seen[k] += 1
for i in sorted(copies):
    c = copies[i]
    match = sum(np.isclose(r["soft_mse"], ref[(r["seed"], r["z_std"])]["soft_mse"], rtol=1e-4)
                for r in c)
    bad = sum(not np.isfinite(r["hard_mse"]) for r in c)
    print(f"copy {i}: n={len(c)} emerged={sum(bool(r['emerged']) for r in c)} "
          f"hard_mse_nonfinite={bad} soft_mse matches uniform_pop512: {match}/{len(c)}")

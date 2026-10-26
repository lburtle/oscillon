import json, numpy as np
rows = [json.loads(l) for l in open("hyperparameters/topo_abl.jsonl")][-150:]
mses = np.array([r["soft_mse"] for r in rows])
print(f"n={len(mses)} median={np.median(mses):.4f} "
      f"IQR=[{np.percentile(mses,25):.4f}, {np.percentile(mses,75):.4f}] "
      f"min={mses.min():.4f} max={mses.max():.4f}")

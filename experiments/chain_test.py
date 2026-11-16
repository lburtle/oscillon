import json, glob, numpy as np
from scipy.stats import spearmanr
recs = [json.load(open(f)) for f in glob.glob("results_topo_abl_v2/rec_seed*_z*.json")]
amp = np.array([r["amp"] for r in recs]); mse = np.array([r["norm_mse"] for r in recs])
print("amp quantiles:", np.percentile(amp, [0, 25, 50, 75, 100]))
print("Spearman(amp, norm_mse):", spearmanr(amp, mse))
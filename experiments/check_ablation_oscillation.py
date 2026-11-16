import json, glob, numpy as np
recs = [json.load(open(f)) for f in glob.glob("results_topo_abl_cycle_v2/rec_seed*_z*.json")]
for name, sel in [("emerged", lambda r: r["emerged"]), ("not emerged", lambda r: not r["emerged"])]:
    rs = [r for r in recs if sel(r) and r["seed_edges_kept"] == 5]
    amp = [r["amp"] for r in rs]
    per = [r["period"] for r in rs if r["period"] is not None]
    print(name, len(rs),
          "| amp median", np.median(amp), "min", min(amp),
          "| period median", np.median(per) if per else None,
          "| norm_mse median", np.median([r["norm_mse"] for r in rs]))
lost = [r for r in recs if r["seed_edges_kept"] < 5]
print("lost an edge:", len(lost), "| norm_mse median", np.median([r["norm_mse"] for r in lost]))

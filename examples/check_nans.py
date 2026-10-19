import json, collections, statistics as st
rows = [json.loads(l) for l in open("results/grad_baseline.jsonl")]
first, second = rows[:100], rows[100:]

print("same (lr, seed) order in both halves:",
      [(r["lr"], r["seed"]) for r in first] == [(r["lr"], r["seed"]) for r in second])

d = [abs(a["final_loss"] - b["final_loss"]) for a, b in zip(first, second)]
print("pairs with identical final_loss:", sum(x == 0 for x in d), "/ 100   max |diff|:", max(d))

by = collections.defaultdict(list)
for r in first: by[r["lr"]].append(r["final_loss"])
for lr in sorted(by):
    v = by[lr]
    print(f"lr={lr:g}  final_loss min/median/max = {min(v):.4g} / {st.median(v):.4g} / {max(v):.4g}")

"""
tally_by_lr.py  <slurm .out file>  <grad_baseline.jsonl>

Per-learning-rate view of the gradient baseline. Pairs, in run order:
  - the k-th training run in the .out  ("grad iter N: loss=X" blocks, each starting at iter 0)
  - the k-th diagnostic dict in the .out (lines starting "{'loss'")
  - the k-th row of the jsonl (gives lr, ridge, emerged, final_loss)
Standard library only.

Handles: blank lines in the jsonl (skipped, counted); NaN/inf losses (the run is counted in a
separate "NaN" column and left out of the medians). If the jsonl has more rows than the .out has
runs (rows left over from an older job), the LAST n rows are used, with a warning.
"""
import ast
import json
import math
import re
import statistics as st
import sys
from collections import defaultdict

IT = re.compile(r"grad iter (\d+): loss=([0-9.eE+-]+|nan|inf|-inf)")

runs, diags = [], []
for line in open(sys.argv[1]):
    m = IT.search(line)
    if m:
        it, loss = int(m.group(1)), float(m.group(2))
        if it == 0:
            runs.append([])
        if runs:
            runs[-1].append((it, loss))
    elif line.startswith("{'loss'"):
        try:
            diags.append(ast.literal_eval(line.strip()))
        except Exception:
            diags.append(None)   # e.g. a dict containing nan; json/ast can't read it

rows, blank, unparsable = [], 0, 0
for l in open(sys.argv[2]):
    if not l.strip():
        blank += 1
        continue
    try:
        rows.append(json.loads(l))   # json.loads accepts NaN / Infinity tokens
    except json.JSONDecodeError:
        unparsable += 1
n = len(runs)
print(f"{n} training runs, {len(diags)} diagnostic dicts, {len(rows)} jsonl rows "
      f"({blank} blank lines skipped, {unparsable} unparsable lines skipped)")
if unparsable:
    print("WARNING: unparsable jsonl lines mean row order may no longer match run order.")
if len(diags) != n:
    sys.exit("run and diagnostic counts differ; cannot pair by order.")
if len(rows) < n:
    sys.exit("fewer jsonl rows than runs; cannot pair by order.")
if len(rows) > n:
    print(f"WARNING: jsonl has {len(rows) - n} extra rows; using the last {n}. "
          f"Check these are the new job's rows (look at the lr values below).")
    rows = rows[-n:]

bad = sum(1 for r, d in zip(runs, diags) if d is not None and abs(round(d["loss"], 4) - r[0][1]) > 1e-4)
none = sum(d is None for d in diags)
print(f"dict loss != iter-0 loss for {bad} runs; {none} dicts unreadable\n")


def klass(d):
    if d is None:
        return "?"
    if d["flat"] and d["grad_norm"] < 1e-6:
        return "plateau"
    if d["grad_norm"] > 1e-3:
        return "live"
    return "in-between"


by_lr = defaultdict(list)
for r, d, row in zip(runs, diags, rows):
    losses = [l for _, l in r]
    finite = all(math.isfinite(l) for l in losses) and math.isfinite(row.get("final_loss", float("nan")))
    start, end = losses[0], losses[-1]
    by_lr[row["lr"]].append({
        "finite": finite, "start": start, "end": end, "drop": start - end,
        "slope": (losses[-1] - losses[-2]) if len(losses) >= 2 else 0.0,
        "class": klass(d), "emerged": bool(row["emerged"]), "ridge": row.get("ridge"),
        "first_nan_iter": next((it for it, l in r if not math.isfinite(l)), None),
    })


def med(xs):
    xs = list(xs)
    return st.median(xs) if xs else float("nan")


print(f"{'lr':>8}{'n':>4}{'NaN':>5}{'med start':>11}{'med end':>9}{'med drop':>10}"
      f"{'>0.02':>7}{'emerged':>9}{'still falling':>15}   (medians/counts over non-NaN runs)")
for lr in sorted(by_lr):
    g = by_lr[lr]
    ok = [x for x in g if x["finite"]]
    print(f"{lr:>8g}{len(g):>4}{len(g) - len(ok):>5}{med(x['start'] for x in ok):>11.4f}"
          f"{med(x['end'] for x in ok):>9.4f}{med(x['drop'] for x in ok):>10.4f}"
          f"{sum(x['drop'] > 0.02 for x in ok):>7}{sum(x['emerged'] for x in g):>9}"
          f"{sum(x['slope'] < -0.001 for x in ok):>15}")

print("\nlive-gradient starts only (grad_norm > 1e-3), non-NaN runs:")
print(f"{'lr':>8}{'n':>4}{'med drop':>10}{'>0.02':>7}")
for lr in sorted(by_lr):
    g = [x for x in by_lr[lr] if x["class"] == "live" and x["finite"]]
    if g:
        print(f"{lr:>8g}{len(g):>4}{med(x['drop'] for x in g):>10.4f}{sum(x['drop'] > 0.02 for x in g):>7}")

nan_runs = [(lr, x) for lr in sorted(by_lr) for x in by_lr[lr] if not x["finite"]]
if nan_runs:
    print(f"\nNaN runs ({len(nan_runs)}):")
    for lr, x in nan_runs:
        where = (f"first non-finite printed loss at iter {x['first_nan_iter']}"
                 if x["first_nan_iter"] is not None else "log finite; final_loss in jsonl is non-finite")
        print(f"  lr={lr:g}  start-class={x['class']}  {where}")

ridges = {x["ridge"] for g in by_lr.values() for x in g}
print(f"\nridge values in the jsonl: {sorted(r for r in ridges if r is not None) or 'not recorded'}")
print("('still falling' = loss decreased by >0.001 over the last 200 printed iterations)")

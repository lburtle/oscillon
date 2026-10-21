"""
tally_progress.py  <slurm .out file>

For each gradient-baseline run in the .out file, compares the loss at iter 0 with the
last printed loss, grouped by the diagnostic class (plateau / live / in-between).
Standard library only.

Assumes the .out has, per run, a block of lines "grad iter N: loss=X" starting at
"grad iter 0", and one diagnostic dict line (starting "{'loss'") per run, in run order.
It checks that assumption: the k-th dict's loss must round to the k-th run's iter-0 loss.
"""
import ast
import re
import statistics as st
import sys

IT = re.compile(r"grad iter (\d+): loss=([0-9.eE+-]+|nan|inf)")

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
            diags.append(None)

print(f"{len(runs)} training runs, {len(diags)} diagnostic dicts")
if len(runs) != len(diags):
    print("counts differ: cannot pair by order. Stop and check the log.")
    sys.exit(1)

bad = sum(1 for r, d in zip(runs, diags) if d is None or abs(round(d["loss"], 4) - r[0][1]) > 1e-4)
print(f"dict loss != iter-0 loss for {bad} runs "
      f"({'diagnostic is at the start params' if bad == 0 else 'diagnostic is NOT (always) at params0, or pairing is off'})")


def klass(d):
    if d["flat"] and d["grad_norm"] < 1e-6:
        return "plateau"
    if d["grad_norm"] > 1e-3:
        return "live"
    return "in-between"


groups = {}
for r, d in zip(runs, diags):
    if d is None:
        continue
    start, end = r[0][1], r[-1][1]
    groups.setdefault(klass(d), []).append((start, end, min(l for _, l in r)))

print(f"\n{'class':<11}{'n':>4}  {'median start':>12}  {'median end':>11}  "
      f"{'median drop':>11}  {'runs dropping >0.02':>20}")
for k in ("plateau", "live", "in-between"):
    g = groups.get(k, [])
    if not g:
        continue
    drops = [s - e for s, e, _ in g]
    print(f"{k:<11}{len(g):>4}  {st.median(s for s, _, _ in g):>12.4f}  "
          f"{st.median(e for _, e, _ in g):>11.4f}  {st.median(drops):>11.4f}  "
          f"{sum(x > 0.02 for x in drops):>20}")
print("\n(start/end = first and last printed loss; the log prints every 200 iters, "
      "so 'end' is iter 2800, not the final iterate)")

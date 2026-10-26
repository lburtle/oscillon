import ast, sys

rows = []
for line in open(sys.argv[1]):
    if line.startswith("{'loss'"):
        try:
            rows.append(ast.literal_eval(line.strip()))
        except Exception:
            print("skipped unparsable line:", line[:80])

n = len(rows)
stuck = sum(r["flat"] and r["grad_norm"] < 1e-6 for r in rows)
live = sum(r["grad_norm"] > 1e-3 for r in rows)
print(n, "runs |", stuck, "on plateau |", live, "with a real gradient")

import json

path = "/scratch/xfd3tf/oscillon/experiments/results/uniform_pop512.json"

def load_runs(path):
    txt = open(path).read()
    try:
        d = json.loads(txt)
        rows = d if isinstance(d, list) else d.get("runs", [d])
    except json.JSONDecodeError:
        rows, bad = [], []
        for i, line in enumerate(txt.splitlines(), 1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                bad.append(i)
        if bad:
            print("unparseable lines:", bad)
    seen, out = set(), []
    for r in rows:
        key = (r.get("seed"), r.get("z_std"))
        seen.add(key)
        out.append(r)
    latest = {(r.get("seed"), r.get("z_std")): r for r in out}
    print(f"{len(rows)} rows, {len(latest)} unique runs")
    return list(latest.values())

load_runs(path)

"""Nhom A #1 (phan chay lap): gop 3 lan chay x 3 cau hinh -> trung binh +- SD mau (n=3).
Chi doc CSV, khong ghi de. FAR/FRR doc tu cot is_abstained GOC (khong tinh lai theo tien to).
Mau so nhu Bang 4.2: Hit@3, FRR tren cau answerable khong loi; citation tren cau answerable
khong tu choi, khong loi; FAR tren cau unanswerable khong loi.
Dung: python script/tuan_bo_sung/aggregate_repeat_runs.py [--dir results/tuan_bo_sung/repeat_runs] [--runs 3]
"""
import csv, argparse, statistics as st
from pathlib import Path

CFGS = ["page_aware", "fixed_size", "longcontext"]
KEYS = ["hit", "cit", "far", "frr"]
LABEL = {"hit": "Hit@3", "cit": "Citation", "far": "FAR", "frr": "FRR"}

def b(v): return str(v).strip().lower() in ("true", "1", "yes")

def metrics(path):
    rows = list(csv.DictReader(open(path, encoding="utf-8")))
    ok = [r for r in rows if not b(r["is_error"])]
    ans = [r for r in ok if b(r["is_answerable"])]
    un = [r for r in ok if not b(r["is_answerable"])]
    m = {"n_rows": len(rows), "n_err": len(rows) - len(ok)}
    m["hit"] = {r["id"]: b(r["hit_at_3"]) for r in ans if str(r.get("hit_at_3", "")).strip() != ""}
    m["cit"] = {r["id"]: b(r["citation_correct"]) for r in ans
                if not b(r["is_abstained"]) and str(r.get("citation_correct", "")).strip() != ""}
    m["far"] = {r["id"]: not b(r["is_abstained"]) for r in un}
    m["frr"] = {r["id"]: b(r["is_abstained"]) for r in ans}
    return m

def pct(d): return 100.0 * sum(d.values()) / len(d) if d else None

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="results/tuan_bo_sung/repeat_runs")
    ap.add_argument("--runs", type=int, default=3)
    a = ap.parse_args()
    R = {c: [metrics(Path(a.dir) / f"{c}_run{i}.csv") for i in range(1, a.runs + 1)] for c in CFGS}
    for c in CFGS:
        print(f"== {c}")
        for i, m in enumerate(R[c], 1):
            cells = []
            for k in KEYS:
                d = m[k]
                cells.append(f"{LABEL[k]} {sum(d.values())}/{len(d)}" if d else f"{LABEL[k]} n/a")
            print(f"  run{i}: rows={m['n_rows']} err={m['n_err']} | " + " | ".join(cells))
        for k in KEYS:
            v = [pct(m[k]) for m in R[c]]
            if any(x is None for x in v): continue
            sd = st.stdev(v) if len(v) > 1 else float("nan")
            print(f"  {LABEL[k]:8s} mean={st.mean(v):.1f}%  SD={sd:.1f}  (cac lan: {', '.join(f'{x:.1f}' for x in v)})")
    print("\n== Cau doi ket qua giua cac lan chay (k = so lan co bien co / tong so lan co gia tri)")
    for c in CFGS:
        for k in KEYS:
            ids = sorted(set().union(*[set(m[k]) for m in R[c]]))
            fl = []
            for i in ids:
                vals = [m[k][i] for m in R[c] if i in m[k]]
                if len(set(vals)) > 1 or len(vals) < len(R[c]):
                    fl.append(f"{i}({sum(vals)}/{len(vals)})")
            if fl: print(f"  {c:11s} {LABEL[k]:8s}: " + ", ".join(fl))
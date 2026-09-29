"""Nhom A #1 (phan khong can API): Wilson 95% CI + McNemar chinh xac tu cac CSV da khoa.
Chi doc file, khong ghi de gi. Doc FAR/FRR tu cot is_abstained GOC (khong tinh lai theo tien to).
Dung: python script/tuan_bo_sung/compute_ci_mcnemar.py [--dir results/tuan8] [--suffix _50cau.csv]
"""
import csv, math, argparse
from pathlib import Path

CFGS = ["page_aware", "fixed_size", "longcontext"]

def b(v): return str(v).strip().lower() in ("true", "1", "yes")

def wilson(k, n, z=1.96):
    if n == 0: return (float("nan"),) * 3
    p = k / n; d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return p, c - h, c + h

def mcnemar_exact(b01, b10):
    n = b01 + b10
    if n == 0: return 1.0
    k = min(b01, b10)
    p = sum(math.comb(n, i) for i in range(0, k + 1)) / 2 ** n * 2
    return min(1.0, p)

def load(d, suffix):
    out = {}
    for c in CFGS:
        rows = list(csv.DictReader(open(Path(d) / f"test_qa_results_{c}{suffix}", encoding="utf-8")))
        out[c] = {r["id"]: r for r in rows}
    return out

def metrics(rows):
    ans = [r for r in rows.values() if b(r["is_answerable"]) and not b(r["is_error"])]
    un = [r for r in rows.values() if not b(r["is_answerable"]) and not b(r["is_error"])]
    hit = {r["id"]: b(r["hit_at_3"]) for r in ans if str(r.get("hit_at_3", "")).strip() != ""}
    cit = {r["id"]: b(r["citation_correct"]) for r in ans
           if not b(r["is_abstained"]) and str(r.get("citation_correct", "")).strip() != ""}
    far = {r["id"]: not b(r["is_abstained"]) for r in un}
    frr = {r["id"]: b(r["is_abstained"]) for r in ans}
    return dict(hit=hit, cit=cit, far=far, frr=frr)

def fmt(k, n):
    p, lo, hi = wilson(k, n)
    return f"{k}/{n} = {p*100:.1f}% [95% CI Wilson {lo*100:.1f}-{hi*100:.1f}]"

def paired(m1, m2, key, ids=None):
    a, c = m1[key], m2[key]
    common = sorted(set(a) & set(c)) if ids is None else sorted(set(ids) & set(a) & set(c))
    b01 = sum(1 for i in common if (not a[i]) and c[i])   # chi cau hinh 2 co bien co
    b10 = sum(1 for i in common if a[i] and (not c[i]))   # chi cau hinh 1 co bien co
    return len(common), b10, b01, mcnemar_exact(b01, b10)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="results/tuan8")
    ap.add_argument("--suffix", default="_50cau.csv")
    a = ap.parse_args()
    data = load(a.dir, a.suffix)
    M = {c: metrics(data[c]) for c in CFGS}
    labels = {"hit": "Hit@3", "cit": "Citation accuracy", "far": "False acceptance (FAR)", "frr": "False refusal (FRR)"}
    for key in ["hit", "cit", "far", "frr"]:
        print(f"== {labels[key]}")
        for c in CFGS:
            d = M[c][key]
            if not d: print(f"  {c:11s}: khong co du lieu"); continue
            print(f"  {c:11s}: {fmt(sum(d.values()), len(d))}")
    print("\n== McNemar chinh xac (so cap, tren cac cau ca hai cau hinh deu co gia tri)")
    pairs = [("page_aware", "fixed_size"), ("page_aware", "longcontext"), ("fixed_size", "longcontext")]
    for key in ["hit", "cit", "far", "frr"]:
        for x, y in pairs:
            if not M[x][key] or not M[y][key]: continue
            n, b10, b01, p = paired(M[x], M[y], key)
            print(f"  {labels[key]:24s} {x} vs {y}: n={n}, chi {x}={b10}, chi {y}={b01}, p={p:.3f}")
    tri = set(M["page_aware"]["cit"]) & set(M["fixed_size"]["cit"]) & set(M["longcontext"]["cit"])
    print(f"\n== Citation tren tap giao 3 cau hinh (n={len(tri)})")
    for c in CFGS:
        k = sum(M[c]["cit"][i] for i in tri); print(f"  {c:11s}: {fmt(k, len(tri))}")
    for x, y in pairs:
        n, b10, b01, p = paired(M[x], M[y], "cit", tri)
        print(f"  McNemar {x} vs {y}: n={n}, {b10}/{b01}, p={p:.3f}")

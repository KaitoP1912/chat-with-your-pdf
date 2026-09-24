"""
so_sanh_2_lan_chay.py - CHI DOC VA IN, khong sua gi, khong can thu vien ngoai.

Muc dich: tra loi chinh xac cau hoi "ngoai test_10 va test_34, con cau nao
doi ket qua giua lan chay Tuan 6 (12/9) va Tuan 8 (20/9) khong?" - vi
Copilot lan truoc tra loi lech sang mot cau hoi khac (danh sach IDs script
tu danh dau "can doi chieu", khong phai diff that giua 2 lan chay).

Cach chay (o thu muc goc du an):
    venv\\Scripts\\python.exe script\\pilot_tuan8\\so_sanh_2_lan_chay.py

Ket qua in ra man hinh VA ghi vao results/tuan8/so_sanh_2_lan_chay.txt
"""
import csv
from pathlib import Path

# File nam o script/pilot_tuan8/, nen goc du an la cha cua cha cua cha thu muc nay
ROOT = Path(__file__).resolve().parent.parent.parent
OUT = []


def p(*a):
    line = " ".join(str(x) for x in a)
    OUT.append(line)
    print(line)


def load(path):
    if not path.exists():
        return None
    with open(path, encoding="utf-8-sig", newline="") as f:
        return {row["id"]: row for row in csv.DictReader(f) if row.get("id")}


# Cap file cu (Tuan 6, 12/9) - moi (Tuan 8, 20/9), theo tung cau hinh
PAIRS = {
    "page_aware": (
        ROOT / "results/tuan6_pilot/test_qa_results_page_aware.csv",
        ROOT / "results/tuan8/test_qa_results_page_aware_50cau.csv",
    ),
    "fixed_size": (
        ROOT / "results/tuan6_pilot/test_qa_results_fixed_size.csv",
        ROOT / "results/tuan8/test_qa_results_fixed_size_50cau.csv",
    ),
    "longcontext": (
        ROOT / "results/tuan6_pilot/test_qa_results_longcontext.csv",
        ROOT / "results/tuan8/test_qa_results_longcontext_50cau.csv",
    ),
}

# Cac cot muon so sanh gia tri THAT giua 2 lan chay
COLUMNS_TO_COMPARE = ["is_abstained", "answer_text", "citations", "citation_correct"]

for config, (old_path, new_path) in PAIRS.items():
    p("=" * 70)
    p(f"CAU HINH: {config}")
    p(f"  cu : {old_path.relative_to(ROOT)}  (ton tai: {old_path.exists()})")
    p(f"  moi: {new_path.relative_to(ROOT)}  (ton tai: {new_path.exists()})")
    old = load(old_path)
    new = load(new_path)
    if old is None or new is None:
        p("  -> BO QUA (thieu 1 trong 2 file)")
        continue

    common_ids = sorted(set(old) & set(new), key=lambda s: (len(s), s))
    p(f"  So cau chung ca 2 file: {len(common_ids)}")
    diff_count = 0
    for qid in common_ids:
        row_old, row_new = old[qid], new[qid]
        diffs = []
        for col in COLUMNS_TO_COMPARE:
            vo, vn = row_old.get(col, ""), row_new.get(col, "")
            if (vo or "").strip() != (vn or "").strip():
                diffs.append(col)
        if diffs:
            diff_count += 1
            p(f"  [{qid}] KHAC O CAC COT: {', '.join(diffs)}")
            for col in diffs:
                vo = (row_old.get(col, "") or "")[:100]
                vn = (row_new.get(col, "") or "")[:100]
                p(f"      cu : {col} = {vo}")
                p(f"      moi: {col} = {vn}")
    if diff_count == 0:
        p("  -> KHONG CO CAU NAO KHAC GIUA 2 LAN CHAY (on dinh tuyet doi)")
    else:
        p(f"  -> TONG SO CAU CO KHAC BIET: {diff_count} / {len(common_ids)}")
    p("")

(ROOT / "results" / "tuan8" / "so_sanh_2_lan_chay.txt").write_text("\n".join(OUT), encoding="utf-8")
print("\n>>> Da ghi results/tuan8/so_sanh_2_lan_chay.txt - dan noi dung file nay cho Claude.")
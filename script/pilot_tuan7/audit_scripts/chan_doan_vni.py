"""
chan_doan_vni_v2.py - Ban sua: dung sys.executable thay vi "python" khi goi
subprocess, dam bao dung dung interpreter cua venv dang active.

Cach dung (dung tai thu muc goc project, venv da active):
    python chan_doan_vni_v2.py
"""
import subprocess
import sys
import re
import os
from pathlib import Path

VNI_FILE = "data/corpus/oldenc_vni_118-2025-qh15_23tr.pdf"
OUT_TXT = Path("results/full_text_inspect/vni_recheck_tuan8.txt")
OUT_TXT.parent.mkdir(parents=True, exist_ok=True)

print(f"Dung Python: {sys.executable}")
print(f"Dang chay check_all.py tren {VNI_FILE} ...")
child_env = os.environ.copy()
child_env["PYTHONIOENCODING"] = "utf-8"
result = subprocess.run(
    [sys.executable, "script/pilot_tuan2/check_all.py", VNI_FILE, str(OUT_TXT)],
    capture_output=True, text=True, encoding='utf-8', errors='replace',
    env=child_env,
)
print("STDOUT (dòng cuối):")
print("\n".join(result.stdout.strip().split("\n")[-15:]))
if result.returncode != 0:
    print("LOI khi chay check_all.py:")
    print(result.stderr[-2000:])
    raise SystemExit(1)

text = OUT_TXT.read_text(encoding="utf-8", errors="replace")

pages = re.split(r"(?:TRANG|Trang|PAGE)\s+(\d+)", text)

suspicious_patterns = [
    r"\bBoû\b", r"\bTheû\b",
    r"[a-zA-Z]\^", r"\(\s*\)",
]

n_pages_with_issue = 0
issue_examples = []
for i in range(1, len(pages), 2):
    page_num = pages[i]
    page_text = pages[i + 1] if i + 1 < len(pages) else ""
    for pat in suspicious_patterns:
        m = re.search(pat, page_text)
        if m:
            n_pages_with_issue += 1
            issue_examples.append((page_num, pat, m.group()))
            break

total_pages_found = (len(pages) - 1) // 2
print(f"\n=== TOM TAT ===")
print(f"Tong so 'trang' nhan dien duoc trong output: {total_pages_found}")
print(f"So trang co dau hieu nghi loi VNI: {n_pages_with_issue}")
if issue_examples:
    print("Vi du cu the:")
    for p, pat, m in issue_examples[:10]:
        print(f"  Trang {p}: mau '{pat}' -> tim thay '{m}'")

print(f"\nFile day du da luu: {OUT_TXT}")
print("LUU Y: day la phat hien tu dong heuristic don gian, van can tu doc lai")
print("file .txt de xac nhan cac trang bi flag co thuc su la loi VNI hay khong.")
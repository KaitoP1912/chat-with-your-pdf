"""
kiem_tra_denominator.py - Kiem tra xem cong thuc tinh answer_correctness
trong aggregate_results.py co dang tinh % tren CA cau unanswerable hay
khong (chi nen tinh tren is_answerable=True).

Cach dung:
    python kiem_tra_denominator.py
(tu ghi ra file ket_qua_denominator.txt, UTF-8)
"""
import csv
from pathlib import Path

OUT_PATH = Path('ket_qua_denominator.txt')
lines = []

for config in ('page_aware', 'fixed_size', 'longcontext'):
    path = Path(f'results/tuan6_pilot/test_qa_results_{config}.csv')
    with open(path, encoding='utf-8') as f:
        rows = list(csv.DictReader(f))

    graded = [r for r in rows if r.get('answer_correctness_manual', '').strip() in ('full', 'partial', 'wrong')]
    graded_answerable = [r for r in graded if r.get('is_answerable', '').strip() == 'True']
    graded_unanswerable = [r for r in graded if r.get('is_answerable', '').strip() != 'True']

    lines.append(f"=== {config} ===")
    lines.append(f"Tong so dong co nhan full/partial/wrong: {len(graded)}")
    lines.append(f"  Trong do answerable=True: {len(graded_answerable)}")
    lines.append(f"  Trong do answerable=False (BI TINH NHAM neu co): {len(graded_unanswerable)}")

    if graded_unanswerable:
        lines.append("  CANH BAO - cac dong unanswerable bi gan nhan full/partial/wrong:")
        for r in graded_unanswerable:
            lines.append(f"    {r['id']}: {r.get('answer_correctness_manual')}")

    def pct(rows_list, label):
        n = len(rows_list)
        if n == 0:
            return f"{label}: n/a"
        full = sum(1 for r in rows_list if r.get('answer_correctness_manual') == 'full')
        partial = sum(1 for r in rows_list if r.get('answer_correctness_manual') == 'partial')
        wrong = sum(1 for r in rows_list if r.get('answer_correctness_manual') == 'wrong')
        return f"{label} (n={n}): full={full/n*100:.1f}% partial={partial/n*100:.1f}% wrong={wrong/n*100:.1f}%"

    lines.append("  " + pct(graded, "Tinh tren TAT CA da cham (co the sai neu lan ca unanswerable)"))
    lines.append("  " + pct(graded_answerable, "Tinh CHI tren answerable=True (dung)"))
    lines.append("")

OUT_PATH.write_text("\n".join(lines), encoding="utf-8")
print(f"Da ghi vao: {OUT_PATH.resolve()}")
print("\n".join(lines))
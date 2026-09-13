"""
fix_grading_manual.py — Sửa tay 3 dòng chấm điểm sai đã phát hiện trong
results/tuan6_pilot/grading_filled_auto.csv, trước khi ghi kết quả vào
3 file CSV chính thức.

3 dòng cần sửa:
  1. (test_10, page_aware): model trả lời "12 kho vận", đáp án đúng là
     "2 kho vận" -> đang bị chấm nhầm "full", sửa thành "wrong".
  2. (test_25, page_aware): câu KHÔNG có đáp án trong tài liệu, nhưng hệ
     thống vẫn trả lời (false acceptance) -> đang bị chấm nhầm "full",
     sửa thành "wrong".
  3. (test_25, fixed_size): giống hệt lý do trên.

Cách chạy (đứng tại thư mục gốc project, D:\\PHUOC\\HK9\\TTTN\\chat-with-your-pdf):
    python fix_grading_manual.py
"""
import csv
from pathlib import Path

path = Path('results/tuan6_pilot/grading_filled_auto.csv')

FIXES = {
    ('test_10', 'page_aware'): (
        'wrong',
        'sua tay: model tra loi "12 kho van", dap an dung la "2 kho van" (sai so lieu)'
    ),
    ('test_25', 'page_aware'): (
        'wrong',
        'sua tay: cau khong co dap an trong tai lieu nhung he thong van tra loi (false acceptance)'
    ),
    ('test_25', 'fixed_size'): (
        'wrong',
        'sua tay: cau khong co dap an trong tai lieu nhung he thong van tra loi (false acceptance)'
    ),
}

with open(path, 'r', encoding='utf-8') as f:
    rows = list(csv.DictReader(f))
    fieldnames = rows[0].keys()

n_fixed = 0
for row in rows:
    key = (row.get('id'), row.get('config'))
    if key in FIXES:
        new_label, note = FIXES[key]
        old_label = row.get('answer_correctness_manual')
        row['answer_correctness_manual'] = new_label
        row['grader_note'] = note
        print(f"Da sua {key}: '{old_label}' -> '{new_label}'")
        n_fixed += 1

with open(path, 'w', newline='', encoding='utf-8') as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rows)

print(f"\nTong so dong da sua: {n_fixed}/{len(FIXES)}")
if n_fixed < len(FIXES):
    print("CANH BAO: co dong khong tim thay trong file — kiem tra lai id/config.")
print(f"Da ghi lai vao: {path}")
print()
print("Buoc tiep theo:")
print("  python script/pilot_tuan6/apply_grading_to_results.py --template results/tuan6_pilot/grading_filled_auto.csv")
print("  python script/pilot_tuan6/aggregate_results.py")
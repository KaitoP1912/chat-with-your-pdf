"""
fix_grading_v3.py — Sửa dòng cuối cùng còn treo: test_20 (fixed_size).
Sau bước này, KHÔNG còn dòng nào trong 75 dòng cần xem lại nữa — dữ liệu
đã sẵn sàng để đưa vào báo cáo Tuần 7 chính thức.

Lý do sửa: câu trả lời đúng ý chính (Mặt trận dân tộc thống nhất phản đế
Đông Dương, đặt nhiệm vụ giải phóng dân tộc lên hàng đầu) nhưng tự thêm 1
câu mâu thuẫn với đáp án mẫu ("Trung ương Đảng... vẫn còn trăn trở, chưa
thật dứt khoát"), trong khi đáp án mẫu khẳng định dứt khoát -> partial,
không phải full.

Cách chạy (đứng tại thư mục gốc project):
    python fix_grading_v3.py
"""
import csv
from pathlib import Path

path = Path('results/tuan6_pilot/grading_filled_auto.csv')

FIXES = {
    ('test_20', 'fixed_size'): (
        'partial',
        'sua tay: dung y chinh nhung tu them cau mau thuan voi dap an mau '
        '("van con tran tro, chua that dut khoat") -> partial'
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
print(f"Da ghi lai vao: {path}")
print()
print("Day la lan sua cuoi cung. Buoc tiep theo:")
print("  python script/pilot_tuan6/apply_grading_to_results.py --template results/tuan6_pilot/grading_filled_auto.csv")
print("  python script/pilot_tuan6/aggregate_results.py")
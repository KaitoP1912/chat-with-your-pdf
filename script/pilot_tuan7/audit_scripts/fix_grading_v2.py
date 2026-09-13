"""
fix_grading_v2.py — Sửa lại theo phản hồi của GVHD (nhận lúc 18h hôm trước):
test_25 phải là "partial" (grounded, citation đúng, nhưng chỉ đáp ứng một
phần phạm vi câu hỏi), KHÔNG PHẢI "wrong" như bản sửa trước.

Lưu ý: cột false_acceptance_rate trong bang_so_sanh_3_cau_hinh.csv KHÔNG cần
sửa gì — nó được tính tự động từ is_abstained/is_answerable, độc lập với cột
answer_correctness_manual này. test_25 vẫn tính đúng là 1 ca false acceptance,
đồng thời vẫn có answer_correctness = partial. Hai chỉ số này tách biệt theo
đúng yêu cầu của thầy.

Cách chạy (đứng tại thư mục gốc project):
    python fix_grading_v2.py
"""
import csv
from pathlib import Path

path = Path('results/tuan6_pilot/grading_filled_auto.csv')

FIXES = {
    ('test_25', 'page_aware'): (
        'partial',
        'sua theo phan hoi GVHD: grounded + citation dung, nhung chi dap ung '
        'mot phan pham vi cau hoi (chi co so lieu 12 ngay dem, khong co tong '
        'ca 2 lan) -> partial, dong thoi van tinh la false acceptance rieng'
    ),
    ('test_25', 'fixed_size'): (
        'partial',
        'sua theo phan hoi GVHD: grounded + citation dung, nhung chi dap ung '
        'mot phan pham vi cau hoi -> partial, dong thoi van tinh la false '
        'acceptance rieng'
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
print("LUU Y: test_10 (page_aware) van giu 'wrong' nhu cu — day la loi so lieu")
print("that (12 kho van vs dap an dung 2 kho van), khong lien quan gi den")
print("phan hoi nay cua thay.")
print()
print("Buoc tiep theo:")
print("  python script/pilot_tuan6/apply_grading_to_results.py --template results/tuan6_pilot/grading_filled_auto.csv")
print("  python script/pilot_tuan6/aggregate_results.py")
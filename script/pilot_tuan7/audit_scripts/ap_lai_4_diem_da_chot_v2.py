"""
ap_lai_4_diem_da_chot_v2.py — Ban sua: KHONG ghi cot grader_note nua (cot
nay khong ton tai trong test_qa_results_*.csv, chi co trong
grading_template.csv/grading_filled_auto.csv). Chi sua dung cot
answer_correctness_manual.

Cach dung:
    python ap_lai_4_diem_da_chot_v2.py
"""
import csv
from pathlib import Path

FIXES = {
    ('page_aware', 'test_10'): 'wrong',
    ('page_aware', 'test_25'): 'partial',
    ('fixed_size', 'test_25'): 'partial',
    ('fixed_size', 'test_20'): 'partial',
}

n_total_fixed = 0

for config in ('page_aware', 'fixed_size', 'longcontext'):
    path = Path(f'results/tuan6_pilot/test_qa_results_{config}.csv')
    with open(path, encoding='utf-8') as f:
        rows = list(csv.DictReader(f))
        fieldnames = list(rows[0].keys())  # giu dung thu tu cot goc, khong them cot moi

    n_fixed_this_file = 0
    for row in rows:
        key = (config, row['id'])
        if key in FIXES:
            new_label = FIXES[key]
            old_label = row.get('answer_correctness_manual')
            row['answer_correctness_manual'] = new_label
            print(f"Da sua {key}: '{old_label}' -> '{new_label}'")
            n_fixed_this_file += 1
            n_total_fixed += 1

    if n_fixed_this_file > 0:
        with open(path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        print(f"  -> Da ghi lai {len(rows)} dong vao {path}")

print(f"\nTong so dong da sua: {n_total_fixed}/4")
print("\nBuoc tiep theo:")
print("  python script/pilot_tuan6/aggregate_results.py")
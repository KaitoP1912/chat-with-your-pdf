#!/usr/bin/env python3
"""
Áp dụng grading từ file template vào 3 file CSV kết quả (page_aware, fixed_size, longcontext).

Cách sử dụng:
  python script/pilot_tuan6/apply_grading_to_results.py --template results/tuan6_pilot/grading_filled_auto.csv
"""
import argparse
import csv
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument('--template', required=True, help='Đường dẫn tới file grading template')
args = parser.parse_args()

# Đọc template
template_path = Path(args.template)
template_rows = {}
with open(template_path, 'r', encoding='utf-8') as f:
    for row in csv.DictReader(f):
        key = (row['id'], row['config'])
        template_rows[key] = row

print(f"Đã tải template từ: {template_path}")
print(f"Tổng dòng trong template: {len(template_rows)}")
print()

# Áp dụng vào 3 file CSV
configs = ['page_aware', 'fixed_size', 'longcontext']
total_updated = 0

for config in configs:
    csv_file = Path(f'results/tuan6_pilot/test_qa_results_{config}.csv')
    if not csv_file.exists():
        print(f"⚠️  {csv_file} không tìm thấy, bỏ qua")
        continue
    
    # Đọc file kết quả
    with open(csv_file, 'r', encoding='utf-8') as f:
        rows = list(csv.DictReader(f))
    
    # Cập nhật grading
    updated = 0
    for row in rows:
        key = (row['id'], config)
        if key in template_rows:
            template = template_rows[key]
            # Cập nhật cột grading
            grade = template.get('answer_correctness_manual', '').strip()
            if grade and grade not in ('abstain', 'skip', 'uncertain'):
                row['answer_correctness_manual'] = grade
                updated += 1
    
    # Ghi lại file
    with open(csv_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    
    print(f"✅ {config:15} | {len(rows):3} dòng | cập nhật {updated:3} dòng")
    total_updated += updated

print()
print(f"Tổng cập nhật: {total_updated} dòng")
print()
print("📝 Tiếp theo:")
print("  1. Chạy: python script/pilot_tuan6/aggregate_results.py")
print("     → để tính lại thống kê với answer_correctness_manual")

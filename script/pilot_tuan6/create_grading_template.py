#!/usr/bin/env python3
"""
Tạo template chấm tay answer_correctness cho 75 dòng (25 câu × 3 cấu hình).

Cách sử dụng:
  1. Chạy script này để tạo file 'grading_template.csv'
  2. Mở file trong Excel, điền cột 'answer_correctness_manual' bằng: full / partial / wrong
  3. Lưu file
  4. Chạy script 'apply_grading_to_results.py' để cập nhật vào 3 file CSV kết quả
"""
import csv
import json
from pathlib import Path

test_qs = json.loads(Path('data/eval_sets/test_questions.json').read_text(encoding='utf-8'))['questions']
test_by_id = {q['id']: q for q in test_qs}

template_rows = []
for config in ['page_aware', 'fixed_size', 'longcontext']:
    csv_file = Path(f'results/tuan6_pilot/test_qa_results_{config}.csv')
    with open(csv_file, 'r', encoding='utf-8') as f:
        rows = list(csv.DictReader(f))
    
    for row in rows:
        # Lấy câu hỏi từ test set
        q_id = row['id']
        q_data = test_by_id.get(q_id, {})
        
        # Chỉ thêm dòng có answer_text (câu đã được trả lời)
        if not row.get('answer_text', '').strip():
            continue
        
        template_rows.append({
            'id': q_id,
            'config': config,
            'question': q_data.get('question', ''),
            'answer_reference': q_data.get('answer_reference', ''),
            'answer_text': row.get('answer_text', ''),
            'is_abstained': row.get('is_abstained', ''),
            'answer_correctness_manual': '',  # Để trống để chấm tay
            'grader_note': '',  # Ghi chú khi chấm
        })

# Ghi ra file
out_path = Path('results/tuan6_pilot/grading_template.csv')
out_path.parent.mkdir(parents=True, exist_ok=True)

fieldnames = [
    'id', 'config', 'question', 'answer_reference', 'answer_text',
    'is_abstained', 'answer_correctness_manual', 'grader_note'
]

with open(out_path, 'w', newline='', encoding='utf-8') as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(template_rows)

print(f"✅ Tạo template chấm tay")
print(f"   - File: {out_path}")
print(f"   - Tổng dòng: {len(template_rows)}")
print()
print("📝 Hướng dẫn:")
print("  1. Mở file grading_template.csv trong Excel")
print("  2. Điền cột 'answer_correctness_manual' bằng 1 trong 3 giá trị:")
print("     - 'full':    đúng hoàn toàn (toàn bộ ý chính, không sai, không bịa)")
print("     - 'partial': đúng một phần (có ý đúng nhưng thiếu hoặc có chi tiết sai)")
print("     - 'wrong':   sai (thông tin chính không khớp hoặc không có trong tài liệu)")
print("  3. (Tùy chọn) Điền cột 'grader_note' để ghi chú ngắn về lý do chấm")
print("  4. Lưu file")
print("  5. Chạy: python script/apply_grading_to_results.py --template results/tuan6_pilot/grading_template.csv")
print()
print("💡 Mẹo:")
print("  - Bỏ qua dòng is_abstained='True' (hệ thống không trả lời)")
print("  - Tập trung vào dòng 'full' vs 'partial' (dòng 'wrong' thường rõ ràng)")

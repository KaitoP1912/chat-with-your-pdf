#!/usr/bin/env python3
"""
Semi-auto grading cho dữ liệu Tuần 8.

Đọc template từ results/tuan8/grading_template_50cau.csv,
gợi ý chấm answer_correctness dựa trên heuristic đơn giản,
riêng cho dữ liệu 50 câu mới.
"""
import csv
import json
import re
from pathlib import Path
from difflib import SequenceMatcher


def strip_markdown(text: str) -> str:
    """Loại bỏ định dạng markdown và các tiền tố mở đầu thường gặp của model."""
    t = text
    t = re.sub(r'^(Dựa trên|Theo)[^:\n]{0,60}[:：]\s*', '', t.strip(), flags=re.IGNORECASE)
    t = re.sub(r'\*\*(.+?)\*\*', r'\1', t)
    t = re.sub(r'\*(.+?)\*', r'\1', t)
    t = re.sub(r'^\s*[-*•]\s+', ' ', t, flags=re.MULTILINE)
    t = re.sub(r'^\s*\d+\.\s+', ' ', t, flags=re.MULTILINE)
    t = re.sub(r'^#{1,6}\s+', '', t, flags=re.MULTILINE)
    t = re.sub(r'\s+', ' ', t).strip()
    return t


def similarity_ratio(a, b):
    """Tính độ tương đồng giữa 2 string (0-1), sau khi đã làm sạch markdown."""
    return SequenceMatcher(None, strip_markdown(a).lower(), strip_markdown(b).lower()).ratio()


def key_facts_overlap(ref: str, text: str) -> float:
    """Đo tỉ lệ các con số/năm/ngày tháng trong answer_reference có xuất hiện lại trong answer_text."""
    ref_numbers = set(re.findall(r'\d[\d.,]*', ref))
    if not ref_numbers:
        return 1.0
    text_clean = strip_markdown(text)
    matched = sum(1 for n in ref_numbers if n in text_clean)
    return matched / len(ref_numbers)


test_qs = json.loads(Path('data/eval_sets/test_questions.json').read_text(encoding='utf-8'))['questions']
test_by_id = {q['id']: q for q in test_qs}

template_path = Path('results/tuan8/grading_template_50cau.csv')
with open(template_path, 'r', encoding='utf-8') as f:
    rows = list(csv.DictReader(f))

for row in rows:
    if row.get('is_abstained') == 'True':
        row['answer_correctness_manual'] = 'abstain'
        row['grader_note'] = 'hệ thống từ chối'
        continue

    answer_ref = row.get('answer_reference', '').strip()
    answer_text = row.get('answer_text', '').strip()

    if not answer_ref or not answer_text:
        row['answer_correctness_manual'] = 'skip'
        row['grader_note'] = 'không có dữ liệu để chấm'
        continue

    sim = similarity_ratio(answer_ref, answer_text)
    fact_overlap = key_facts_overlap(answer_ref, answer_text)

    if sim > 0.55 or fact_overlap >= 0.9:
        row['answer_correctness_manual'] = 'full'
        row['grader_note'] = f'similarity={sim:.2f}, fact_overlap={fact_overlap:.2f} (cao)'
    elif sim > 0.35 or fact_overlap >= 0.5:
        row['answer_correctness_manual'] = 'partial'
        row['grader_note'] = f'similarity={sim:.2f}, fact_overlap={fact_overlap:.2f}, cần verify chi tiết'
    else:
        row['answer_correctness_manual'] = 'uncertain'
        row['grader_note'] = f'similarity={sim:.2f}, fact_overlap={fact_overlap:.2f}, cần đọc tay'

out_path = Path('results/tuan8/grading_filled_auto_50cau.csv')
out_path.parent.mkdir(parents=True, exist_ok=True)

with open(out_path, 'w', newline='', encoding='utf-8') as f:
    writer = csv.DictWriter(f, fieldnames=rows[0].keys())
    writer.writeheader()
    writer.writerows(rows)

stats = {}
for row in rows:
    label = row.get('answer_correctness_manual', 'unknown')
    stats[label] = stats.get(label, 0) + 1

print(f"✅ Semi-auto grading Tuần 8 hoàn tất")
print(f"   - File: {out_path}")
print()
print("Phân bổ:")
for label, count in sorted(stats.items(), key=lambda x: -x[1]):
    print(f"  {label:12}: {count:3} dòng")
print()
print("⚠️  Đây vẫn chỉ là gợi ý sơ bộ; cần đọc lại các dòng partial / uncertain.")

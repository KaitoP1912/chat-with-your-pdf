#!/usr/bin/env python3
"""Kiểm tra số lượng câu trong dev/test set."""
import json
from pathlib import Path

root = Path(__file__).parent.parent

dev = json.loads((root / 'data/eval_sets/dev_questions_normalized.json').read_text(encoding='utf-8'))['questions']
test = json.loads((root / 'data/eval_sets/test_questions.json').read_text(encoding='utf-8'))['questions']

dev_ids = {q['id'] for q in dev}
test_ids = {q['id'] for q in test}

print(f'Dev set: {len(dev)} câu')
print(f'Test set: {len(test)} câu')
print(f'Overlap: {len(dev_ids & test_ids)} câu (nên là 0)')
print(f'Cần thêm test: {50 - len(test)} câu')
print(f'Tổng corpus hiện có: {len(dev_ids | test_ids)} câu')
print()
print(f'Danh sách id dev không có trong test (top 10):')
for qid in sorted(dev_ids - test_ids)[:10]:
    print(f'  - {qid}')

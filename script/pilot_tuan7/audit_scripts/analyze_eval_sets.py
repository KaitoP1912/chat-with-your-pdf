#!/usr/bin/env python3
"""Phân tích cân bằng dev/test set."""
import json
from pathlib import Path

root = Path('.')
dev = json.loads((root / 'data/eval_sets/dev_questions_normalized.json').read_text(encoding='utf-8'))['questions']
test = json.loads((root / 'data/eval_sets/test_questions.json').read_text(encoding='utf-8'))['questions']

dev_ids = {q['id'] for q in dev}
test_ids = {q['id'] for q in test}
only_in_dev_ids = dev_ids - test_ids

dev_answerable = [q for q in dev if q.get('is_answerable')]
dev_unanswerable = [q for q in dev if not q.get('is_answerable')]
test_answerable = [q for q in test if q.get('is_answerable')]
test_unanswerable = [q for q in test if not q.get('is_answerable')]

only_in_dev = [q for q in dev if q['id'] in only_in_dev_ids]
only_in_dev_answerable = [q for q in only_in_dev if q.get('is_answerable')]
only_in_dev_unanswerable = [q for q in only_in_dev if not q.get('is_answerable')]
only_in_dev_bridge = [q for q in only_in_dev_answerable if q.get('is_bridge_case')]

print('DEV SET:')
print(f'  Answerable: {len(dev_answerable)}')
print(f'  Unanswerable: {len(dev_unanswerable)}')
print()
print('TEST SET (hiện tại 25):')
print(f'  Answerable: {len(test_answerable)} (bridge: {len([q for q in test_answerable if q.get("is_bridge_case")])})')
print(f'  Unanswerable: {len(test_unanswerable)}')
print()
print('CHỈ TRONG DEV (25 câu cần thêm):')
print(f'  Tổng: {len(only_in_dev)}')
print(f'  Answerable: {len(only_in_dev_answerable)} (bridge: {len(only_in_dev_bridge)})')
print(f'  Unanswerable: {len(only_in_dev_unanswerable)}')
print()
print('ĐỀ XU:')
print(f'Giữ nguyên cân bằng test set hiện tại:')
print(f'  17 answerable (2 bridge): thêm từ dev: {17 - len(test_answerable)}')
print(f'  8 unanswerable: thêm từ dev: {8 - len(test_unanswerable)}')
print()
print('ID câu mới cần thêm (25 câu đầu tiên từ only_in_dev):')
for i, q in enumerate(sorted(only_in_dev, key=lambda x: x['id'])[:25]):
    bridge_marker = ' [BRIDGE]' if q.get('is_bridge_case') else ''
    answerable_type = 'answerable' if q.get('is_answerable') else f"unanswerable ({q.get('type')})"
    print(f'  {i+1:2}. {q["id"]:8} {answerable_type:20} {bridge_marker}')

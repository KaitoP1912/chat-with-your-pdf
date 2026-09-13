"""
gop_50_cau.py — Gop 25 cau moi (da verify xong, trong cau_hoi_moi_25.json)
vao test_questions.json hien co (25 cau cu, test_01 - test_25), danh so
ID moi la test_26 - test_50. Xoa cac truong noi bo (_status) khong can
trong file cuoi cung.

TRUOC KHI CHAY: backup file test_questions.json cu, phong khi can khoi
phuc:
    Copy-Item data/eval_sets/test_questions.json data/eval_sets/test_questions_25cau_backup.json

Cach dung:
    python gop_50_cau.py
"""
import json
from pathlib import Path

TEST_PATH = Path('data/eval_sets/test_questions.json')
NEW_PATH = Path('cau_hoi_moi_25.json')

old_data = json.loads(TEST_PATH.read_text(encoding='utf-8'))
old_questions = old_data['questions'] if isinstance(old_data, dict) and 'questions' in old_data else old_data

new_questions_raw = json.loads(NEW_PATH.read_text(encoding='utf-8'))
new_questions_raw = new_questions_raw['questions'] if isinstance(new_questions_raw, dict) and 'questions' in new_questions_raw else new_questions_raw

if len(old_questions) != 25:
    print(f"CANH BAO: file cu dang co {len(old_questions)} cau, khong phai 25. Dung lai kiem tra.")
    raise SystemExit(1)

if len(new_questions_raw) != 25:
    print(f"CANH BAO: file moi dang co {len(new_questions_raw)} cau, khong phai 25. Dung lai kiem tra.")
    raise SystemExit(1)

merged = list(old_questions)

for i, q in enumerate(new_questions_raw, start=26):
    q_clean = {k: v for k, v in q.items() if not k.startswith('_')}
    q_clean['id'] = f'test_{i}'
    merged.append(q_clean)

output = {'questions': merged} if isinstance(old_data, dict) and 'questions' in old_data else merged

TEST_PATH.write_text(
    json.dumps(output, ensure_ascii=False, indent=2),
    encoding='utf-8'
)

print(f"Da gop xong: {len(merged)} cau (test_01 - test_50)")
print(f"Da ghi vao: {TEST_PATH.resolve()}")
print()
print("Buoc tiep theo:")
print("  1. Kiem tra lai: python -c \"import json; d=json.load(open('data/eval_sets/test_questions.json',encoding='utf-8')); print(len(d['questions']))\"")
print("  2. Chay lai pipeline danh gia tren ca 50 cau (3 cau hinh)")
print("  3. Cham tay cau_correctness cho 25 cau moi (test_26-test_50)")
print("  4. Chay lai aggregate_results.py")
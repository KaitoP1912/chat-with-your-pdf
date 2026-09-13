"""
sua_dinh_dang_expected_page.py — Chuan hoa expected_page trong
test_questions.json: luon dung dang list, ke ca cau 1 trang (vi du
10 -> [10]), giu nguyen None cho cau unanswerable, giu nguyen list
cho cau bridge.

Cach dung:
    python sua_dinh_dang_expected_page.py
"""
import json
from pathlib import Path

TEST_PATH = Path('data/eval_sets/test_questions.json')

data = json.loads(TEST_PATH.read_text(encoding='utf-8'))
questions = data['questions'] if isinstance(data, dict) and 'questions' in data else data

n_fixed = 0
for q in questions:
    p = q.get('expected_page')
    if p is None:
        continue
    if isinstance(p, int):
        q['expected_page'] = [p]
        n_fixed += 1
    elif isinstance(p, list):
        pass  # da dung dinh dang
    else:
        print(f"CANH BAO: cau {q.get('id')} co expected_page kieu la ({type(p)}): {p}")

output = {'questions': questions} if isinstance(data, dict) and 'questions' in data else questions
TEST_PATH.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding='utf-8')

print(f"Da sua {n_fixed} cau tu so nguyen tran sang dang list.")
print(f"Da ghi lai vao: {TEST_PATH.resolve()}")

# Kiem tra lai ngay: khong con cau nao la int tran
data2 = json.loads(TEST_PATH.read_text(encoding='utf-8'))
questions2 = data2['questions'] if isinstance(data2, dict) and 'questions' in data2 else data2
con_loi = [q['id'] for q in questions2 if isinstance(q.get('expected_page'), int)]
if con_loi:
    print(f"VAN CON LOI o cac cau: {con_loi}")
else:
    print("Xac nhan: khong con cau nao expected_page la so nguyen tran.")
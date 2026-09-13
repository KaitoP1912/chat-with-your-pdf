import json

def load(path):
    return json.load(open(path, encoding='utf-8'))['questions']

dev = load('data/eval_sets/dev_questions_normalized.json')
test = load('data/eval_sets/test_questions.json')

print(f"Dev set: {len(dev)} cau | Test set hien tai: {len(test)} cau")
print()
print("=" * 70)
print("VUNG CAM - khong duoc dat cau hoi moi trung trang/chu de nay")
print("=" * 70)

banned = []
for q in dev + test:
    pages = q.get('expected_page') or q.get('answer_pages') or []
    banned.append({
        'id': q['id'],
        'source_file': q.get('source_file'),
        'pages': pages,
        'question': q['question'][:80],
    })

for b in sorted(banned, key=lambda x: (x['source_file'] or '', str(x['pages']))):
    print(f"  [{b['source_file']}] trang {b['pages']} - {b['id']}: {b['question']}")

print()
print("=" * 70)
print("TOM TAT THEO FILE - trang da bi 'dung' , TRANH khi soan cau moi")
print("=" * 70)
by_file = {}
for b in banned:
    by_file.setdefault(b['source_file'], set())
    for p in (b['pages'] if isinstance(b['pages'], list) else [b['pages']]):
        if p is not None:
            by_file[b['source_file']].add(p)

for fname, pages in by_file.items():
    print(f"  {fname}: trang {sorted(pages)}")

# Ghi ra file de dung lam input cho buoc soan cau hoi moi
with open('vung_cam_trang_da_dung.json', 'w', encoding='utf-8') as f:
    json.dump({k: sorted(v) for k, v in by_file.items()}, f, ensure_ascii=False, indent=2)
print()
print("Da ghi danh sach vung cam vao: vung_cam_trang_da_dung.json")
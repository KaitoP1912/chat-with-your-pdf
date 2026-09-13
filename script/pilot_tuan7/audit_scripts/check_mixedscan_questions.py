import json

for fname in ['data/eval_sets/test_questions.json', 'data/eval_sets/dev_questions_normalized.json']:
    try:
        d = json.load(open(fname, encoding='utf-8'))
    except FileNotFoundError:
        print(f'{fname} khong ton tai')
        continue

    qs = d['questions']
    print(f'--- {fname} ({len(qs)} cau) ---')
    print('Cac truong (key) co trong 1 cau hoi mau:', list(qs[0].keys()))
    print()

    # Liet ke toan bo gia tri source_file/file/pdf_file... de xem dung ten truong nao
    for key_candidate in ['source_file', 'file', 'pdf_file', 'document', 'source']:
        if key_candidate in qs[0]:
            values = sorted(set(q.get(key_candidate, '') for q in qs))
            print(f'Truong "{key_candidate}" - cac gia tri xuat hien:')
            for v in values:
                print(' ', v)
            print()

    # Tim rieng nhung cau co nhac "qcvn" hoac "mixedscan" o BAT KY truong nao (dang string)
    print('Cac cau co nhac "qcvn" hoac "mixedscan" o bat ky truong nao:')
    found = False
    for q in qs:
        text_blob = json.dumps(q, ensure_ascii=False).lower()
        if 'qcvn' in text_blob or 'mixedscan' in text_blob:
            print(' ', q.get('id'), '->', json.dumps(q, ensure_ascii=False)[:300])
            found = True
    if not found:
        print('  (khong tim thay cau nao)')
    print()
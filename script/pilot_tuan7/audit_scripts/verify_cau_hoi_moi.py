"""
verify_cau_hoi_moi.py — Kiem tra 25 cau hoi moi (Gemini vua soan) co that su
khong dung vao vung cam hay khong, TRUOC KHI dua vao test_questions.json.

Cach dung:
1. Luu output JSON cua Gemini vao file: cau_hoi_moi_25.json
   (dinh dang: {"questions": [...]} hoac list truc tiep [...])
2. Chay: python verify_cau_hoi_moi.py
3. Chi tiep tuc neu thay dong "TAT CA 25 CAU DEU AN TOAN"
"""
import json
from pathlib import Path

BANNED = json.loads(Path('vung_cam_trang_da_dung.json').read_text(encoding='utf-8'))

raw = json.loads(Path('cau_hoi_moi_25.json').read_text(encoding='utf-8'))
new_questions = raw['questions'] if isinstance(raw, dict) and 'questions' in raw else raw

print(f"Tong so cau moi doc duoc: {len(new_questions)}")
if len(new_questions) != 25:
    print(f"!!! CANH BAO: khong dung 25 cau (dang co {len(new_questions)}). Kiem tra lai file JSON.")

errors = []
seen_pages_this_batch = {}  # kiem tra luon trung lap NOI BO trong 25 cau moi

for i, q in enumerate(new_questions):
    source = q.get('source_file', '')
    pages = q.get('expected_page')
    if pages is None:
        pages_list = []
    elif isinstance(pages, list):
        pages_list = pages
    else:
        pages_list = [pages]

    banned_pages_for_file = set(BANNED.get(source, []))

    # Kiem tra 1: co dung trang cam khong
    hit_banned = [p for p in pages_list if p in banned_pages_for_file]
    if hit_banned:
        errors.append(
            f"  [LOI] Cau {i+1} ({source}, trang {pages_list}): "
            f"DUNG TRANG CAM {hit_banned} -> LOAI BO cau nay, phai soan lai"
        )

    # Kiem tra 2: bridge case co dung 2 trang lien ke khong
    if q.get('is_bridge_case') and len(pages_list) == 2:
        p1, p2 = sorted(pages_list)
        if p2 - p1 != 1:
            errors.append(
                f"  [LOI] Cau {i+1} ({source}): bridge case nhung 2 trang {pages_list} "
                f"KHONG lien ke nhau -> khong hop le"
            )

    # Kiem tra 3: trung lap voi CHINH cau khac trong 25 cau moi nay
    key = (source, tuple(sorted(pages_list)))
    if key in seen_pages_this_batch and pages_list:
        errors.append(
            f"  [LOI] Cau {i+1} trung trang voi cau {seen_pages_this_batch[key]+1} "
            f"trong CHINH 25 cau moi nay ({source}, trang {pages_list})"
        )
    seen_pages_this_batch[key] = i

    # Kiem tra 4: cau answerable ma khong co answer_reference/expected_page -> nghi ngo
    if q.get('is_answerable') and (not q.get('answer_reference') or not pages_list):
        errors.append(
            f"  [LOI] Cau {i+1} ({source}): danh dau is_answerable=True nhung "
            f"thieu answer_reference hoac expected_page"
        )

print()
if errors:
    print(f"PHAT HIEN {len(errors)} LOI - KHONG duoc dung cac cau nay:")
    for e in errors:
        print(e)
    print()
    print("=> Sua lai hoac soan thay the cac cau bi loi, chay lai script nay cho den khi sach.")
else:
    print("TAT CA 25 CAU DEU AN TOAN - khong dung vung cam, khong trung lap noi bo.")
    print("Van con 1 buoc BAT BUOC truoc khi dung: TU DOC LAI TUNG CAU, doi chieu")
    print("answer_reference voi dung trang trong PDF that (Gemini co the trich sai/bia).")
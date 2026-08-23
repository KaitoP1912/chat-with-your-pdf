"""
Chuẩn hóa schema cho dev_questions.json (34 câu).

Vấn đề: các câu unanswerable mới thêm (out_of_scope mới + toàn bộ near_miss)
đang thiếu field so với câu answerable / out_of_scope cũ:
  - answer_reference
  - answer_pages
  - is_bridge_case

Script này KHÔNG sửa nội dung câu hỏi/đáp án đã có, chỉ bổ sung field còn
thiếu với giá trị mặc định hợp lý, để toàn bộ 34 câu có cùng schema.

Cách chạy:
    python normalize_dev_questions.py dev_questions.json dev_questions_normalized.json
"""
import json
import sys


REQUIRED_DEFAULTS = {
    "answer_reference": "KHÔNG CÓ THÔNG TIN TRONG TÀI LIỆU",
    "answer_pages": [],
    "is_bridge_case": False,
    "out_of_scope": None,   # sẽ được set true/false bên dưới, không để None lọt ra ngoài
    "type": "answerable",   # cho các câu answerable chưa có field type
}


def normalize_question(q: dict) -> dict:
    q = dict(q)  # copy, không sửa object gốc

    is_answerable = q.get("is_answerable", True)

    # answer_reference
    if "answer_reference" not in q:
        q["answer_reference"] = REQUIRED_DEFAULTS["answer_reference"]

    # answer_pages
    if "answer_pages" not in q:
        q["answer_pages"] = []

    # is_bridge_case
    if "is_bridge_case" not in q:
        q["is_bridge_case"] = False

    # out_of_scope (bool) — chỉ có ý nghĩa cho loại out_of_scope,
    # near_miss vẫn unanswerable nhưng KHÔNG phải out_of_scope
    if "out_of_scope" not in q:
        q["out_of_scope"] = (not is_answerable) and q.get("type") == "out_of_scope"

    # type — câu answerable cũ (dev_01..dev_14, dev_16..dev_25) chưa có field này
    if "type" not in q:
        q["type"] = "answerable" if is_answerable else "unknown_unanswerable_type"

    # related_page — chỉ near_miss mới có; thêm null cho các loại khác để
    # đồng nhất key (không bắt buộc, nhưng tránh KeyError khi code check .get)
    if "related_page" not in q:
        q["related_page"] = None

    # note — near_miss/out_of_scope mới có; thêm rỗng cho câu answerable
    if "note" not in q:
        q["note"] = ""

    return q


def main():
    if len(sys.argv) != 3:
        print("Cách dùng: python normalize_dev_questions.py <input.json> <output.json>")
        sys.exit(1)

    in_path, out_path = sys.argv[1], sys.argv[2]

    with open(in_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    data["questions"] = [normalize_question(q) for q in data["questions"]]

    # Kiểm tra nhanh: tất cả câu phải có cùng 1 bộ key
    key_sets = {frozenset(q.keys()) for q in data["questions"]}
    if len(key_sets) > 1:
        print("[CẢNH BÁO] Vẫn còn lệch schema giữa các câu sau khi chuẩn hóa:")
        for ks in key_sets:
            print("  -", sorted(ks))
    else:
        print(f"[OK] Toàn bộ {len(data['questions'])} câu đã đồng nhất schema:")
        print("  Keys:", sorted(next(iter(key_sets))))

    # Thống kê nhanh để đối chiếu với _ghi_chu
    n_answerable = sum(1 for q in data["questions"] if q["is_answerable"])
    n_unanswerable = len(data["questions"]) - n_answerable
    n_bridge = sum(1 for q in data["questions"] if q.get("is_bridge_case"))
    n_out_of_scope = sum(1 for q in data["questions"] if q.get("type") == "out_of_scope")
    n_near_miss = sum(1 for q in data["questions"] if q.get("type") == "near_miss")

    print(f"\n  Tổng số câu       : {len(data['questions'])}")
    print(f"  Answerable        : {n_answerable} (trong đó bridge: {n_bridge})")
    print(f"  Unanswerable      : {n_unanswerable}")
    print(f"    - out_of_scope  : {n_out_of_scope}")
    print(f"    - near_miss     : {n_near_miss}")

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    print(f"\nĐã ghi file chuẩn hóa: {out_path}")


if __name__ == "__main__":
    main()
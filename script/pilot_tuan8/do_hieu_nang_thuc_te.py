"""
do_hieu_nang_thuc_te.py - Do thoi gian xu ly thuc te (ingestion + hoi dap),
KHONG dung de danh gia do chinh xac, KHONG dung de so sanh 3 cau hinh.

Muc dich: tra loi cau hoi "tu luc tai PDF len den luc hoi duoc, nguoi dung
phai cho bao lau?" - con so nay hien KHONG co trong Bang 4.2 (Bang 4.2 chi
do thoi gian hoi-dap, khong do thoi gian dung chi muc).

AN TOAN: chi doc file corpus co san, KHONG sua config, KHONG ghi de bat ky
file ket qua Tuan 6/8 nao. Ket qua luu rieng vao results/trai_nghiem_thuc_te/.

Cach chay (o thu muc goc du an):
    venv\\Scripts\\python.exe script\\pilot_tuan8\\do_hieu_nang_thuc_te.py

QUAN TRONG: chay script nay 1 MINH, dung mo dong thoi voi Streamlit,
de khong bi may cham vi chay 2 tien trinh nang cung luc, gay so lieu sai.
"""
import csv
import sys
import time
from pathlib import Path

# File nam o script/pilot_tuan8/, nen goc du an la cha cua cha cua cha thu muc nay
ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

OUT_DIR = ROOT / "results" / "trai_nghiem_thuc_te"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Chon vai file dai dien, tu nho den lon, de thay xu huong theo so trang.
# Sua lai duong dan neu ten file cua ban khac.
FILES = [
    ROOT / "data/corpus/nodiacritic_118-2025-qh15_20tr.pdf",   # nho
    ROOT / "data/corpus/normal_hienphap_33tr.pdf",              # vua
    ROOT / "data/corpus/normal_vinamilkbaocao2014_53tr.pdf",    # lon
    ROOT / "data/corpus/normal_lichsudang_C1&2_60tr.pdf",       # lon nhat
]

# Vai cau hoi ngan, chi de do thoi gian hoi-dap tren tung tai lieu da dung
# san index - KHONG dung de cham diem dung/sai.
SAMPLE_QUESTIONS = {
    "normal_hienphap_33tr.pdf": "Điều 61 nói gì về giáo dục?",
    "normal_vinamilkbaocao2014_53tr.pdf": "Vinamilk có bao nhiêu nhà máy?",
}

VNCORENLP_DIR = str(ROOT / "vncorenlp_models")

rows = []


def timed(label, fn, *args, **kwargs):
    t0 = time.perf_counter()
    result = fn(*args, **kwargs)
    dt = time.perf_counter() - t0
    print(f"    {label}: {dt:.2f}s")
    return result, dt


def main():
    from source.ingestion.pdf_loader import load_pdf_pages
    from source.ingestion.scan_detector import detect_scan
    from source.retrieval.ingest_glue import build_clean_pages
    from source.retrieval.chunker import chunk_by_page
    from source.retrieval.vectorstore import build_index, search

    for i, f in enumerate(FILES, 1):
        if not f.exists():
            print(f"[BO QUA] khong tim thay {f}")
            continue

        print(f"\n=== Tai lieu {i}/{len(FILES)}: {f.name} ===")
        note = "CO TAI MO HINH (lan dau)" if i == 1 else "khong tai lai mo hinh"
        print(f"  ({note})")

        t_start = time.perf_counter()
        pages, t_load = timed("Doc PDF (load_pdf_pages)", load_pdf_pages, str(f))
        n_pages = len(pages)
        _, t_scan = timed("Phat hien scan (detect_scan)", detect_scan, pages)
        clean_pages, t_clean = timed("Chuan hoa (build_clean_pages)", build_clean_pages, str(f))
        chunks, t_chunk = timed("Chia doan (chunk_by_page)", chunk_by_page, clean_pages)
        index, t_index = timed(
            "Dung chi muc (build_index)", build_index, chunks, VNCORENLP_DIR
        )
        t_total_ingest = time.perf_counter() - t_start

        print(f"  -> TONG THOI GIAN SAN SANG DE HOI: {t_total_ingest:.2f}s "
              f"cho {n_pages} trang, {len(chunks)} chunk")

        row = {
            "file": f.name,
            "so_trang": n_pages,
            "so_chunk": len(chunks),
            "lan_chay_thu": i,
            "co_tai_mo_hinh_lan_dau": (i == 1),
            "t_doc_pdf_s": round(t_load, 2),
            "t_phat_hien_scan_s": round(t_scan, 2),
            "t_chuan_hoa_s": round(t_clean, 2),
            "t_chia_doan_s": round(t_chunk, 2),
            "t_dung_chi_muc_s": round(t_index, 2),
            "TONG_thoi_gian_san_sang_s": round(t_total_ingest, 2),
        }

        # Neu co cau hoi mau cho file nay, do luon thoi gian hoi-dap
        q = SAMPLE_QUESTIONS.get(f.name)
        if q:
            t0 = time.perf_counter()
            hits = search(index, q, VNCORENLP_DIR, k=15)
            t_search = time.perf_counter() - t0
            print(f"    Tim kiem cho cau '{q[:40]}...': {t_search:.2f}s "
                  f"(chi do buoc tim kiem, chua goi Gemini)")
            row["cau_hoi_mau"] = q
            row["t_tim_kiem_s"] = round(t_search, 2)

        rows.append(row)

    if not rows:
        print("\nKhong co file nao chay duoc, kiem tra lai duong dan FILES.")
        return

    out_csv = OUT_DIR / "hieu_nang_thuc_te.csv"
    fieldnames = [
        "file", "so_trang", "so_chunk", "lan_chay_thu", "co_tai_mo_hinh_lan_dau",
        "t_doc_pdf_s", "t_phat_hien_scan_s", "t_chuan_hoa_s", "t_chia_doan_s",
        "t_dung_chi_muc_s", "TONG_thoi_gian_san_sang_s",
        "cau_hoi_mau", "t_tim_kiem_s",
    ]
    with open(out_csv, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            row.setdefault("cau_hoi_mau", "")
            row.setdefault("t_tim_kiem_s", "")
            writer.writerow(row)

    print(f"\n>>> Da ghi ket qua vao {out_csv.relative_to(ROOT)}")
    print(">>> File nay KHONG lien quan gi den results/tuan6_pilot hay results/tuan8")
    print(">>> Dan noi dung hieu_nang_thuc_te.csv cho Claude neu can phan tich them.")


if __name__ == "__main__":
    main()
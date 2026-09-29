import csv
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from source.retrieval.ingest_glue import build_clean_pages


CORPUS_DIR = ROOT / "data" / "corpus"
OUTPUT_PATH = ROOT / "results" / "trai_nghiem_thuc_te" / "longcontext_timing.csv"


def main() -> None:
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    if OUTPUT_PATH.exists():
        raise SystemExit(f"Tu choi ghi de file ket qua da ton tai: {OUTPUT_PATH}")

    rows = []
    files = sorted(path for path in CORPUS_DIR.iterdir() if path.is_file())

    for path in files:
        if path.name.casefold() == "test2.pdf":
            print(f"[BO QUA] {path.name}")
            continue
        if path.suffix.casefold() != ".pdf":
            print(f"[BO QUA - khong ho tro PDF] {path.name}")
            continue

        started = time.perf_counter()
        try:
            clean_pages = build_clean_pages(str(path))
        except Exception as exc:
            elapsed = time.perf_counter() - started
            print(f"{path.name}: loi sau {elapsed:.3f} giay - {exc}")
            rows.append({
                "file": path.name,
                "so_trang": "",
                "thoi_gian_giay": round(elapsed, 3),
                "trang_thai": "loi",
                "loi": str(exc),
            })
            continue

        elapsed = time.perf_counter() - started
        print(f"{path.name}: {elapsed:.3f} giay ({len(clean_pages)} trang)")
        rows.append({
            "file": path.name,
            "so_trang": len(clean_pages),
            "thoi_gian_giay": round(elapsed, 3),
            "trang_thai": "thanh_cong",
            "loi": "",
        })

    fieldnames = ["file", "so_trang", "thoi_gian_giay", "trang_thai", "loi"]
    with OUTPUT_PATH.open("x", newline="", encoding="utf-8-sig") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Da luu ket qua: {OUTPUT_PATH.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
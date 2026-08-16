"""
encoding_evaluation.py

Script đo lường Precision, Recall, F1 (cho nhãn encoding), CER và False Conversion Rate.
Input: CSV ground-truth với các cột: id, raw_text, expected_label, expected_normalized
 - expected_label: original, tcvn3, vni, unknown
 - expected_normalized: phiên bản Unicode chuẩn đã kiểm chứng độc lập để tính CER

Usage:
    python script/pilot_tuan2/encoding_evaluation.py script/ground_truth_manual_verified.csv
"""
from __future__ import annotations

import csv
import os
import sys
from collections import Counter
from typing import Dict, List

# Sửa đường dẫn lùi 2 cấp thư mục (.. , ..) để trỏ chính xác về thư mục gốc của project
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from source.ingestion.text_normalizer import normalize_page_text


def levenshtein(a: str, b: str) -> int:
    """Tính khoảng cách Levenshtein giữa hai chuỗi."""
    if a == b:
        return 0
    la, lb = len(a), len(b)
    if la == 0:
        return lb
    if lb == 0:
        return la
    d = [[0] * (lb + 1) for _ in range(la + 1)]
    for i in range(la + 1):
        d[i][0] = i
    for j in range(lb + 1):
        d[0][j] = j
    for i in range(1, la + 1):
        for j in range(1, lb + 1):
            cost = 0 if a[i - 1] == b[j - 1] else 1
            d[i][j] = min(d[i - 1][j] + 1, d[i][j - 1] + 1, d[i - 1][j - 1] + cost)
    return d[la][lb]


def char_error_rate(a: str, b: str) -> float:
    """Tính Character Error Rate (CER)."""
    if not a and not b:
        return 0.0
    dist = levenshtein(a, b)
    denom = max(1, len(b))
    return dist / denom


def compute_cer(a: str, b: str) -> float:
    """Wrapper tương thích với các automated regression tests."""
    return char_error_rate(a, b)


def compute_metrics(rows: List[Dict[str, str]]):
    y_true = []
    y_pred = []
    cer_all = []
    cer_on_correct = []
    per_row = []
    false_conversion_count = 0
    total = 0

    for r in rows:
        raw_text = r["raw_text"]
        expected_label = r["expected_label"].strip().lower()
        expected_norm = r.get("expected_normalized", "")

        res = normalize_page_text(raw_text)
        pred_label = res.encoding_decision.strip().lower()
        pred_norm = res.normalized_text

        y_true.append(expected_label)
        y_pred.append(pred_label)

        cer = char_error_rate(pred_norm, expected_norm)
        cer_all.append(cer)

        # Tính CER trên các trường hợp nhận diện đúng nhãn
        if pred_label == expected_label:
            cer_on_correct.append(cer)

        # Kiểm tra False Conversion: văn bản gốc chuẩn hoặc unknown nhưng bị ép convert sang tcvn3/vni
        if expected_label in {"original", "unknown"} and pred_label in {"tcvn3", "vni"}:
            false_conversion_count += 1

        per_row.append({
            "id": r.get("id", ""),
            "expected_label": expected_label,
            "pred_label": pred_label,
            "cer": cer,
        })

        total += 1

    classes = sorted(set(y_true) | set(y_pred))
    tp = Counter()
    fp = Counter()
    fn = Counter()
    for t, p in zip(y_true, y_pred):
        if t == p:
            tp[t] += 1
        else:
            fp[p] += 1
            fn[t] += 1

    metrics = {}
    for c in classes:
        prec = tp[c] / (tp[c] + fp[c]) if (tp[c] + fp[c]) > 0 else 0.0
        rec = tp[c] / (tp[c] + fn[c]) if (tp[c] + fn[c]) > 0 else 0.0
        f1 = (2 * prec * rec) / (prec + rec) if (prec + rec) > 0 else 0.0
        metrics[c] = (prec, rec, f1)

    avg_cer_all = sum(cer_all) / len(cer_all) if cer_all else 0.0
    avg_cer_correct = sum(cer_on_correct) / len(cer_on_correct) if cer_on_correct else 0.0
    false_conversion_rate = false_conversion_count / total if total else 0.0

    print("=== Classification metrics (label-level)")
    print("Label,Precision,Recall,F1")
    for c in classes:
        prec, rec, f1 = metrics[c]
        print(f"{c},{prec:.3f},{rec:.3f},{f1:.3f}")
    print()

    print("=== Character-level conversion quality (CER)")
    print(f"Avg CER (all rows): {avg_cer_all:.3f}")
    print(f"Avg CER (only correctly classified rows): {avg_cer_correct:.3f} (n={len(cer_on_correct)})")
    print(f"False Conversion Rate: {false_conversion_rate:.3f} ({false_conversion_count}/{total})")
    print()

    print("=== Per-row CER details (id, expected_label, pred_label, CER)")
    for r in per_row:
        print(f"{r['id']},{r['expected_label']},{r['pred_label']},{r['cer']:.3f}")


def load_csv(path: str) -> List[Dict[str, str]]:
    with open(path, newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        return [row for row in reader]


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python script/pilot_tuan2/encoding_evaluation.py script/ground_truth_manual_verified.csv")
        sys.exit(1)
    rows = load_csv(sys.argv[1])
    compute_metrics(rows)
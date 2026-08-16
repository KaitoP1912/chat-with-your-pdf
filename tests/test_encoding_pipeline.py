import csv
from pathlib import Path
import pytest
from source.ingestion.text_normalizer import (
    normalize_page_text,
    convert_tcvn3_to_unicode,
    fix_missing_u_horn,
)
from source.ingestion.pdf_loader import load_pdf_pages

GROUND_TRUTH_PATH = Path('script/ground_truth_manual_verified.csv')
MAX_CER_THRESHOLD = 0.05


def load_ground_truth():
    with open(GROUND_TRUTH_PATH, encoding='utf-8') as f:
        return list(csv.DictReader(f))


def char_error_rate(hyp: str, ref: str) -> float:
    try:
        from script.pilot_tuan2.encoding_evaluation import compute_cer
        return compute_cer(hyp, ref)
    except Exception:
        if not ref:
            return 0.0 if not hyp else 1.0
        import difflib
        matcher = difflib.SequenceMatcher(None, ref, hyp)
        return 1.0 - matcher.ratio()


@pytest.mark.parametrize('row', load_ground_truth())
def test_classification_correct(row):
    result = normalize_page_text(row['raw_text'])
    assert result.encoding_decision == row['expected_label'], (
        f"{row['id']}: expected {row['expected_label']}, got {result.encoding_decision}"
    )


@pytest.mark.parametrize('row', load_ground_truth())
def test_character_level_accuracy(row):
    result = normalize_page_text(row['raw_text'])
    if result.encoding_decision == row['expected_label']:
        cer = char_error_rate(result.normalized_text, row['expected_normalized'])
        assert cer <= MAX_CER_THRESHOLD, f"{row['id']}: CER={cer:.3f} vượt ngưỡng {MAX_CER_THRESHOLD}"


def test_no_false_conversion_on_clean_corpus():
    clean_files = [
        'normal_hienphap_33tr.pdf',
        'normal_vinamilkbaocao2014_53tr.pdf',
        'normal_lichsudang_C1&2_60tr.pdf',
        'nodiacritic_118-2025-qh15_20tr.pdf',
    ]
    total_checked_pages = 0
    for filename in clean_files:
        p = Path('data/corpus') / filename
        assert p.exists(), f"LỖI: File corpus {filename} không tồn tại trong data/corpus/!"

        pages = load_pdf_pages(str(p))
        assert len(pages) > 0, f"LỖI: File {filename} không đọc được trang nào!"

        for page in pages:
            result = normalize_page_text(page.raw_text)
            assert result.encoding_decision in ('original', 'unknown'), (
                f"{filename} trang {page.page_number}: bị chuyển nhầm thành {result.encoding_decision}"
            )
            total_checked_pages += 1

    print(f"\n[XÁC NHẬN KIỂM THỬ THẬT] Đã quét qua toàn bộ {total_checked_pages} trang sạch mà không bị False Conversion.")


def test_hyphen_not_corrupted():
    cases = [
        'Số điện thoại: 0123-456-789',
        'Ngày 01-02-2024 là lễ.',
        'Danh sách: - Mục 1 - Mục 2 - Mục 3',
        'Ghi chú: giảm 10%-20% so với năm trước.',
        'Điều 1 - Phạm vi điều chỉnh',
    ]
    for text in cases:
        result = normalize_page_text(text)
        assert '-' in result.normalized_text


def test_missing_u_horn_and_vni_mojibake_regression():
    res_vni = normalize_page_text('Traàn Thanh Mẫn')
    assert 'Trần Thanh Mẫn' == res_vni.normalized_text or 'Traàn' not in res_vni.normalized_text

    res_u_horn = normalize_page_text('Xe đ ợc đưa đón theo đ ờng quy định.')
    assert 'được' in res_u_horn.normalized_text
    assert 'đường' in res_u_horn.normalized_text


def test_tcvn3_dash_as_u_horn_isolated():
    """
    Kiểm tra thuật toán giải mã TCVN3 khôi phục đúng dấu '-' thay cho 'ư'.
    Test trực tiếp tầng chuyển đổi bảng mã (convert_tcvn3_to_unicode + fix_missing_u_horn).
    """
    test_cases = [
        ("ph-¬ng tiÖn", "phương tiện"),
        ("ng-êi tham gia", "người tham gia"),
        ("®-êng bé", "đường bộ"),
        ("®-a ra ®Êu gi¸", "đưa ra đấu giá"),
    ]
    for raw, expected in test_cases:
        converted = convert_tcvn3_to_unicode(raw)
        result = fix_missing_u_horn(converted)
        assert expected in result, (
            f"Lỗi: '{raw}' không chuyển thành '{expected}', nhận được: '{result}'"
        )
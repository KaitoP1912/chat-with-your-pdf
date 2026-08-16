"""
text_normalizer.py — Nhóm 3 (Tuần 2 & Tuần 3 Refactored)

Nhiệm vụ:
    3a. Suy luận và chuẩn hóa văn bản mã cũ (TCVN3, VNI-Windows) -> Unicode NFC.
        Module thử các ánh xạ ứng viên (original, TCVN3, VNI) và chấm điểm bằng từ điển tiếng Việt
        kết hợp tỷ lệ nguyên âm có dấu.
    3b. Phát hiện văn bản bị mất dấu tiếng Việt dựa trên mật độ nguyên âm có dấu.
    3c. Loại bỏ số trang rác và số trang dính liền vào đầu dòng nội dung.
    3d. Khôi phục glyph "ư" bị rơi trong các PDF lỗi font dựa trên quy luật âm vị học
        (phụ âm đầu hợp lệ + khoảng trắng hoặc dấu '-' + nguyên âm).
"""

from __future__ import annotations

import csv
import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

# --------------------------------------------------------------------------
# 3c. LỌC BỎ SỐ TRANG IN & SỐ TRANG DÍNH LIỀN ĐẦU DÒNG
# --------------------------------------------------------------------------

def remove_page_number_artifacts(text: str) -> str:
    """
    Loại bỏ các dòng chứa số trang rác hoặc số trang dính liền vào đầu dòng nội dung.
    Ví dụ: 'Trang 12', '12 / 60', '3 2. Các dân tộc...' -> '2. Các dân tộc...'
    """
    if not text:
        return ""

    lines = text.split("\n")
    cleaned_lines = []

    page_standalone_pattern = re.compile(
        r"^\s*(trang|page)?\s*[\-\–\—]?\s*\d+(\s*[\/\-]\s*\d+)?\s*[\-\–\—]?\s*$",
        re.IGNORECASE
    )
    embedded_header_pattern = re.compile(r"^\s*\d+\s+([A-ZĐƯÊÔÀÁẢÃẠa-zđưêôàáảãạ].*)$")

    for idx, line in enumerate(lines):
        line_str = line.strip()
        is_edge_line = (idx < 2) or (idx >= len(lines) - 2)

        if is_edge_line:
            if page_standalone_pattern.match(line_str):
                continue
            match = embedded_header_pattern.match(line)
            if match:
                cleaned_lines.append(match.group(1))
                continue

        cleaned_lines.append(line)

    return "\n".join(cleaned_lines).strip()


# --------------------------------------------------------------------------
# 3d. KHÔI PHỤC KÝ TỰ "ư" BỊ RỚT (lỗi glyph dấu cách hoặc dấu '-' trong PDF)
# --------------------------------------------------------------------------

_VIETNAMESE_ONSETS = [
    "ngh", "kh", "ph", "th", "tr", "ch", "nh", "gi", "gh", "ng", "qu",
    "b", "c", "d", "đ", "g", "h", "k", "l", "m", "n", "p", "r", "s", "t", "v", "x",
]
_VIETNAMESE_VOWELS_LOWER = (
    "aàáảãạăằắẳẵặâầấẩẫậeèéẻẽẹêềếểễệiìíỉĩịoòóỏõọôồốổỗộơờớởỡợ"
    "uùúủũụưừứửữựyỳýỷỹỵ"
)

# Khôi phục chữ 'ư' khi bị rớt thành dấu cách (space)
_missing_u_horn_space_pattern = re.compile(
    r"(?<![A-Za-zÀ-ỹ])(" + "|".join(_VIETNAMESE_ONSETS) + r")[ \t]+"
    r"(?=[" + _VIETNAMESE_VOWELS_LOWER + r"])",
    re.IGNORECASE,
)

# Khôi phục chữ 'ư' khi font PDF trích xuất nhầm thành dấu gạch nối '-' (ví dụ: ph-ơng, ng-ời, đ-ờng, đ-a)
_missing_u_horn_dash_pattern = re.compile(
    r"(?<![A-Za-zÀ-ỹ0-9])(" + "|".join(_VIETNAMESE_ONSETS) + r")-"
    r"(?=[" + _VIETNAMESE_VOWELS_LOWER + r"])",
    re.IGNORECASE,
)


def _restore_all_caps_context(text: str) -> str:
    """Khôi phục chữ hoa cho token đứng ở giữa cụm từ viết hoa toàn bộ."""
    if not text:
        return text

    parts = re.split(r"(\s+)", text)
    for i, part in enumerate(parts):
        if not part or part.isspace():
            continue
        if not re.fullmatch(r"[A-Za-zÀ-ỹ]+", part):
            continue
        if len(part) <= 1:
            continue
        prev = parts[i - 1] if i > 0 else ""
        nxt = parts[i + 1] if i + 1 < len(parts) else ""
        if prev and nxt and prev.isupper() and nxt.isupper():
            parts[i] = part.upper()
    return "".join(parts)


def fix_missing_u_horn(text: str) -> str:
    """
    Chèn lại ký tự 'ư' tại các vị trí glyph bị rơi (thành khoảng trắng hoặc dấu '-').
    Áp dụng hoàn toàn theo quy luật âm vị học tiếng Việt, bảo toàn 100% dấu gạch nối thật.
    """
    if not text:
        return text

    def _replace(match: re.Match) -> str:
        return match.group(1) + "ư"

    text = _missing_u_horn_space_pattern.sub(_replace, text)
    text = _missing_u_horn_dash_pattern.sub(_replace, text)

    return _restore_all_caps_context(text)


# --------------------------------------------------------------------------
# 3a. SUY LUẬN BẢNG MÃ CŨ -> UNICODE
# --------------------------------------------------------------------------

CONFIDENCE_THRESHOLD: float = 0.50  # Tối thiểu để chấp nhận chuyển đổi
MIN_IMPROVEMENT: float = 0.12       # Cải thiện tối thiểu so với original
AMBIGUOUS_MARGIN: float = 0.05      # Nếu gần nhau -> ambiguous

WORDMATCH_WEIGHT: float = 0.80      # Trọng số điểm khớp từ điển
ACCENT_WEIGHT: float = 0.20         # Trọng số mật độ nguyên âm có dấu

# Bảng mã TCVN 5712:1993 chuẩn
TCVN3_MAP: Dict[str, str] = {
    "µ": "à", "¸": "á", "¶": "ả", "·": "ã", "¹": "ạ",
    "¨": "ă", "»": "ằ", "¾": "ắ", "¼": "ẳ", "½": "ẵ", "Æ": "ặ",
    "©": "â", "Ç": "ầ", "Ê": "ấ", "É": "ẩ", "È": "ẫ", "Ë": "ậ",
    "Ì": "è", "Ð": "é", "Î": "ẻ", "Ü": "ĩ", "Ö": "ệ",
    "ª": "ê", "Ò": "ề", "Õ": "ế", "Ó": "ể", "Ô": "ễ",
    "«": "ô", "å": "ồ", "è": "ố", "æ": "ổ", "ç": "ỗ", "é": "ộ",
    "¬": "ơ", "ê": "ờ", "í": "ớ", "ë": "ở", "ì": "ỡ", "î": "ợ",
    "ï": "ù", "ó": "ú", "ñ": "ủ", "ü": "ũ", "ô": "ụ",
    "ð": "ừ", "ú": "ứ", "ø": "ứ", "ö": "ử", "÷": "ữ", "ù": "ự",
    "×": "ì", "Ý": "í", "Ø": "ỉ", "Þ": "ị",
    "ß": "ò", "ã": "ó", "á": "ỏ", "õ": "õ", "ä": "ọ",
    "®": "đ", "§": "Đ",
}

# Bảng mã VNI-Windows chuẩn (hỗ trợ đầy đủ tổ hợp 3 ký tự và chữ hoa)
_VNI_BASE: Dict[str, str] = {
    "ieàu": "iều", "ieäu": "iệu", "ieåu": "iểu", "ieãu": "iễu", "ieáu": "iếu",
    "aù": "á", "aø": "à", "aoû": "ả", "aõ": "ã", "aï": "ạ",
    "aé": "ắ", "aè": "ằ", "aú": "ẳ", "aü": "ẵ", "aë": "ặ", "ađ": "ă",
    "aá": "ấ", "aầ": "ầ", "aẩ": "ẩ", "aẫ": "ẫ", "aậ": "ậ", "aâ": "â",
    "eù": "é", "eø": "è", "eoû": "ẻ", "eõ": "ẽ", "eï": "ẹ",
    "eá": "ế", "eầ": "ề", "eẩ": "ể", "eẫ": "ễ", "eậ": "ệ", "eđ": "ê",
    "où": "ó", "oø": "ò", "ooû": "ỏ", "oõ": "õ", "oï": "ọ",
    "oá": "ố", "oầ": "ồ", "oẩ": "ổ", "oẫ": "ỗ", "oậ": "ộ", "ođ": "ô",
    "ôù": "ớ", "ôø": "ờ", "ôû": "ở", "ôõ": "ỡ", "ôï": "ợ", "ô": "ơ",
    "uù": "ú", "uø": "ù", "uoû": "ủ", "uõ": "ũ", "uï": "ụ",
    "öù": "ứ", "öø": "ừ", "öû": "ử", "öõ": "ữ", "öï": "ự", "ö": "ư",
    "yù": "ý", "yø": "ỳ", "yoû": "ỷ", "yõ": "ỹ", "yï": "ỵ",
    "aê": "ă", "eä": "ệ", "oâ": "ô", "uû": "ủ",
    "aå": "ẩ", "aû": "ả", "aä": "ậ", "eà": "ề", "eâ": "ê", "oà": "ồ", "oä": "ộ", "oå": "ổ", "oã": "ỗ",
    "aã": "ẫ", "ó": "ĩ",
    "iaà": "ià", "ieå": "iể", "ieã": "iễ", "ieà": "iề", "eå": "ể", "eủ": "ẻ",
    "oủ": "ỏ", "uaà": "uà", "aàu": "ầu", "aàn": "ần", "aà": "à",
    "æ": "ỉ", "ñ": "đ", "Ñ": "Đ", "ò": "ị", "Û": "Ủ", "UÛ": "Ủ",
    "đ": "đ", "Đ": "Đ",
}

VNI_MAP: Dict[str, str] = dict(_VNI_BASE)
for _k, _v in _VNI_BASE.items():
    _uk = _k.upper()
    _uv = _v.upper()
    if _uk not in VNI_MAP:
        VNI_MAP[_uk] = _uv

ENCODING_MAPS: Dict[str, Dict[str, str]] = {
    "tcvn3": TCVN3_MAP,
    "vni": VNI_MAP,
}


@dataclass
class EncodingDetectionResult:
    detected_encoding: Optional[str]
    confidence: float
    normalized_text: str
    ambiguous: bool
    warning: Optional[str] = None


def default_wordlist() -> Set[str]:
    """
    Tải bộ từ vựng chuẩn tiếng Việt (hư từ ngữ pháp + từ vựng hành chính/pháp lý tổng quát).
    Tuyệt đối không chứa các từ vựng trích xuất từ tập kiểm thử.
    """
    function_words = {
        "và", "của", "là", "có", "trong", "cho", "được", "này", "các", "một",
        "những", "mỗi", "mọi", "tất", "cả", "toàn", "bộ", "riêng", "chung", "cùng",
        "khác", "nhau", "rất", "quá", "còn", "vẫn", "đang", "sẽ", "đã", "phải",
        "cần", "không", "chưa", "chỉ", "cũng", "lại", "nữa", "thêm", "hơn", "nhất",
        "khi", "với", "để", "như", "sau", "trước", "trên", "dưới", "giữa", "từ",
        "đến", "vào", "ra", "lên", "xuống", "nếu", "thì", "mà", "nhưng", "vì",
        "nên", "do", "bởi", "tại", "ở", "đó", "kia", "ấy", "theo", "về",
        "hoặc", "đã", "ai", "gì", "sao", "nào", "đâu", "làm", "việc", "vậy",
        "được", "bị", "cho", "rằng", "là", "hay", "còn", "nữa", "thế", "nào",
    }
    domain_words = {
        "người", "nước", "việt", "nam", "hiến", "pháp", "luật", "quy", "định",
        "điều", "khoản", "chính", "phủ", "nhà", "công", "dân", "xã", "hội",
        "kinh", "tế", "văn", "hóa", "giáo", "dục", "thực", "hiện", "sự", "nghiệp",
        "trách", "nhiệm", "quyền", "nghĩa", "vụ",
        "chủ", "cộng", "hòa", "độc", "lập", "tự", "do", "hạnh", "phúc",
        "quốc", "hội", "báo", "cáo", "tịch", "ban", "hành", "nghị",
        "thông", "tư", "quyết", "tổ", "chức", "cơ", "quan", "quản", "lý",
        "phát", "triển", "xây", "dựng", "đầu", "tư", "doanh", "nghiệp", "sản", "xuất",
        "thương", "mại", "dịch", "tài", "chính", "ngân", "hàng", "chi", "phí",
        "thu", "lợi", "nhuận", "kế", "hoạch", "chương", "trình", "dự", "án",
        "đảng", "trung", "ương", "lịch", "sử", "đấu", "tranh", "cách", "mạng",
        "giai", "đoạn", "thời", "kỳ", "chiến", "thắng", "nhân", "quân", "đội",
        "số", "lượng", "mức", "tỷ", "triệu", "nghìn", "phần", "trăm", "năm",
        "tháng", "ngày", "giờ", "vấn", "đề", "kết", "quả", "nội", "dung",
        "thông", "tin", "mục", "trang", "tài", "liệu", "hồ", "sơ", "sản", "phẩm",
        "khách", "hàng", "thị", "trường", "lao", "động", "nhân", "viên", "sức", "khỏe",
    }
    base = function_words | domain_words

    source_dir = Path(__file__).resolve().parents[1]
    external_path = source_dir / "vietnamese_wordlist_external.txt"

    if external_path.exists():
        try:
            with external_path.open(encoding="utf-8") as fh:
                extras = {line.strip().lower() for line in fh if line.strip()}
            return base | extras
        except Exception:
            pass

    repo_root = Path(__file__).resolve().parents[2]
    rdr_path = repo_root / "vncorenlp_models" / "models" / "wordsegmenter" / "wordsegmenter.rdr"
    if rdr_path.exists():
        try:
            content = rdr_path.read_text(encoding="utf-8")
            tokens = set(re.findall(r'"([^"\n]+)"', content))
            filtered = {t.lower().strip() for t in tokens if len(t.strip()) > 1 and any(ch.isalpha() for ch in t)}
            return base | filtered
        except Exception:
            pass

    return base


def _dict_match_ratio(text: str, wordlist: Set[str]) -> float:
    words = re.findall(r"[^\W\d_]+", text.lower(), flags=re.UNICODE)
    if not words:
        return 0.0
    matched = sum(1 for w in words if w in wordlist)
    return matched / len(words)


def _accent_coverage_ratio(text: str) -> float:
    letters = [ch for ch in text if ch.isalpha()]
    if not letters:
        return 0.0
    vietnamese_diacritical_chars = set("àáảãạăằắẳẵặâầấẩẫậèéẻẽẹêềếểễệìíỉĩịòóỏõọôồốổỗộơờớởỡợùúủũụưừứửữựỳýỷỹỵđ")
    return sum(1 for ch in letters if ch.lower() in vietnamese_diacritical_chars) / len(letters)


def _looks_all_caps_word(raw_token: str) -> bool:
    letters = [ch for ch in raw_token if ch.isalpha() or ch in {"§", "Ñ"}]
    if not letters:
        return False
    if any("a" <= ch <= "z" for ch in raw_token):
        return False
    uppercase_count = sum(1 for ch in raw_token if "A" <= ch <= "Z")
    return uppercase_count >= 2 or "§" in raw_token or "Ñ" in raw_token


def _restore_all_caps_words(raw_text: str, normalized_text: str) -> str:
    raw_parts = re.split(r"(\s+)", raw_text)
    norm_parts = re.split(r"(\s+)", normalized_text)
    restored_parts: List[str] = []

    for idx, raw_part in enumerate(raw_parts):
        if idx >= len(norm_parts):
            break
        norm_part = norm_parts[idx]
        if raw_part.isspace():
            restored_parts.append(norm_part)
            continue
        if _looks_all_caps_word(raw_part):
            restored_parts.append(norm_part.upper())
        else:
            restored_parts.append(norm_part)

    if len(norm_parts) > len(raw_parts):
        restored_parts.extend(norm_parts[len(raw_parts):])

    return "".join(restored_parts)


def _try_decode_with_map(text: str, char_map: Dict[str, str]) -> str:
    if not text or not char_map:
        return text

    sorted_keys = sorted(char_map.keys(), key=len, reverse=True)
    pattern = re.compile("|".join(re.escape(k) for k in sorted_keys))

    def replace_match(match: re.Match) -> str:
        return char_map[match.group(0)]

    return pattern.sub(replace_match, text)


def convert_tcvn3_to_unicode(text: str) -> str:
    return _try_decode_with_map(text, TCVN3_MAP)


def convert_vni_to_unicode(text: str) -> str:
    return _try_decode_with_map(text, VNI_MAP)


def _normalize_soft_hyphens(text: str) -> str:
    if not text:
        return text
    text = text.replace("\u00AD", "")
    text = re.sub(r"-\n\s*", "", text)
    text = re.sub(r"\n-\s*", "\n", text)
    text = re.sub(r"[ \t]{2,}", " ", text)
    return text


def _score_candidate_text(raw_text: str, text: str, wordlist: Set[str]) -> Tuple[str, float]:
    cleaned = _normalize_soft_hyphens(text)
    normalized = unicodedata.normalize("NFC", cleaned)
    normalized = _restore_all_caps_words(raw_text, normalized)

    dict_score = _dict_match_ratio(normalized, wordlist)
    accent_score = _accent_coverage_ratio(normalized)

    score = (WORDMATCH_WEIGHT * dict_score) + (ACCENT_WEIGHT * accent_score)

    words = re.findall(r"[^\W\d_]+", normalized, flags=re.UNICODE)
    if len(words) <= 2 and dict_score == 0.0:
        return normalized, accent_score * ACCENT_WEIGHT

    return normalized, score


def normalize_encoding(
    text: str,
    wordlist: Optional[Set[str]] = None,
    ambiguous_margin: float = 0.05,
) -> EncodingDetectionResult:
    wordlist = wordlist or default_wordlist()
    candidates: List[Tuple[Optional[str], str, float]] = []

    nfc_original = unicodedata.normalize("NFC", text)
    nfc_original = fix_missing_u_horn(nfc_original)
    nfc_original, original_score = _score_candidate_text(text, nfc_original, wordlist)
    candidates.append((None, nfc_original, original_score))

    for enc_name, char_map in ENCODING_MAPS.items():
        converted = _try_decode_with_map(text, char_map)
        converted = fix_missing_u_horn(converted)
        converted, score = _score_candidate_text(text, converted, wordlist)
        if converted == nfc_original:
            continue
        candidates.append((enc_name, converted, score))

    candidates.sort(key=lambda c: c[2], reverse=True)
    best_enc, best_text, best_score = candidates[0]
    second_score = candidates[1][2] if len(candidates) > 1 else 0.0

    ambiguous = (best_score - second_score) < ambiguous_margin
    warning = None
    if ambiguous and best_enc is not None:
        warning = f"Không đủ tin cậy chọn bảng mã (Score max={best_score:.2f}, 2nd={second_score:.2f})."

    return EncodingDetectionResult(
        detected_encoding=best_enc,
        confidence=best_score,
        normalized_text=best_text,
        ambiguous=ambiguous,
        warning=warning,
    )


# --------------------------------------------------------------------------
# 3b. PHÁT HIỆN MẤT DẤU
# --------------------------------------------------------------------------

@dataclass
class DiacriticCheckResult:
    dict_match_ratio: float
    likely_missing_diacritics: bool


def _vietnamese_diacritic_ratio(text: str) -> float:
    vietnamese_diacritical_chars = set("àáảãạăằắẳẵặâầấẩẫậèéẻẽẹêềếểễệìíỉĩịòóỏõọôồốổỗộơờớởỡợùúủũụưừứửữựỳýỷỹỵđĐ")
    letters = [ch.lower() for ch in text if ch.isalpha()]
    if not letters:
        return 0.0
    diacritic_count = sum(1 for ch in letters if ch in vietnamese_diacritical_chars)
    return diacritic_count / len(letters)


def detect_missing_diacritics(
    text: str,
    wordlist: Optional[Set[str]] = None,
    threshold: float = 0.03,
) -> DiacriticCheckResult:
    ratio = _vietnamese_diacritic_ratio(text)
    return DiacriticCheckResult(
        dict_match_ratio=ratio,
        likely_missing_diacritics=ratio < threshold,
    )


# --------------------------------------------------------------------------
# HÀM TỔNG HỢP CHO TỪNG TRANG
# --------------------------------------------------------------------------

@dataclass
class NormalizationResult:
    normalized_text: str
    detected_encoding: Optional[str]
    encoding_confidence: float
    encoding_warning: Optional[str]
    likely_missing_diacritics: bool
    encoding_decision: str = "unknown"


def normalize_page_text(raw_text: str, wordlist: Optional[Set[str]] = None) -> NormalizationResult:
    wordlist = wordlist or default_wordlist()
    had_escaped_newlines = "\\n" in raw_text and "\n" not in raw_text
    processing_raw_text = raw_text.replace("\\n", "\n") if had_escaped_newlines else raw_text

    # 1. Cắt bỏ số trang rác / Header / Footer dính đầu dòng
    text_no_artifacts = remove_page_number_artifacts(processing_raw_text)

    # 2. Chuẩn bị base cho chấm điểm: loại soft-hyphen trước khi chấm
    scoring_base = _normalize_soft_hyphens(text_no_artifacts)
    scoring_base = unicodedata.normalize("NFC", scoring_base)

    # 3. Tạo 3 phiên bản song song: original, as_tcvn3, as_vni
    original_candidate_raw = fix_missing_u_horn(scoring_base)
    original_candidate, original_score = _score_candidate_text(text_no_artifacts, original_candidate_raw, wordlist)

    tcvn_candidate_raw = convert_tcvn3_to_unicode(scoring_base)
    tcvn_candidate_raw = fix_missing_u_horn(tcvn_candidate_raw)
    tcvn_candidate, tcvn_score = _score_candidate_text(text_no_artifacts, tcvn_candidate_raw, wordlist)

    vni_candidate_raw = convert_vni_to_unicode(scoring_base)
    vni_candidate_raw = fix_missing_u_horn(vni_candidate_raw)
    vni_candidate, vni_score = _score_candidate_text(text_no_artifacts, vni_candidate_raw, wordlist)

    candidates = [
        ("original", original_candidate, original_score),
        ("tcvn3", tcvn_candidate, tcvn_score),
        ("vni", vni_candidate, vni_score),
    ]

    candidates.sort(key=lambda c: c[2], reverse=True)
    best_name, best_text, best_score = candidates[0]
    second_score = candidates[1][2]

    encoding_warning = None
    ambiguous = (best_score - second_score) < AMBIGUOUS_MARGIN
    if ambiguous:
        encoding_warning = f"Ứng viên đầu và thứ hai sát nhau (best={best_score:.2f}, 2nd={second_score:.2f})."

    # 4. Quyết định chọn bảng mã theo logic tối ưu
    encoding_decision = "unknown"
    detected_encoding: Optional[str] = None

    if best_name != "original":
        diff_with_original = best_score - original_score
        if best_score >= CONFIDENCE_THRESHOLD and diff_with_original >= MIN_IMPROVEMENT and not ambiguous:
            encoding_decision = best_name
            detected_encoding = best_name
        else:
            encoding_decision = "unknown"
    else:
        if original_score >= CONFIDENCE_THRESHOLD:
            encoding_decision = "original"
            detected_encoding = None
        else:
            encoding_decision = "unknown"

    # 5. Áp dụng chuyển đổi theo quyết định
    if encoding_decision == "tcvn3":
        candidate_text = tcvn_candidate
    elif encoding_decision == "vni":
        candidate_text = vni_candidate
    else:
        candidate_text = unicodedata.normalize("NFC", original_candidate_raw)

    # 6. Sửa lỗi glyph 'ư' rớt độc lập (dấu cách hoặc dấu '-')
    final_text = fix_missing_u_horn(candidate_text)

    # 7. Phát hiện mất dấu dựa trên văn bản cuối cùng
    diac_result = detect_missing_diacritics(final_text, wordlist=wordlist)
    if had_escaped_newlines:
        final_text = final_text.replace("\n", "\\n")

    return NormalizationResult(
        normalized_text=final_text,
        detected_encoding=detected_encoding,
        encoding_confidence=best_score,
        encoding_warning=encoding_warning,
        likely_missing_diacritics=diac_result.likely_missing_diacritics,
        encoding_decision=encoding_decision,
    )
"""
text_normalizer.py — Nhóm 3 (Tuần 2 - Cleaned & Finalized)

Nhiệm vụ:
    3a. Suy luận và chuẩn hóa văn bản mã cũ (TCVN3, VNI-Windows) -> Unicode NFC.
            Lưu ý: đây không phải bài toán "thử decode rồi chọn kết quả nhìn hợp lệ";
            module thử các ánh xạ ứng viên và chấm điểm bằng từ điển tiếng Việt.
            Hai file kiểm thử TCVN3/VNI trong tuần 2 là controlled test cases được
            tạo từ Unicode gốc bằng UniKey (Ctrl + Shift + F6), không phải tài liệu
            lỗi thu thập từ nguồn bên ngoài.
    3b. Phát hiện văn bản bị mất dấu tiếng Việt dựa trên mật độ nguyên âm có dấu.
    3c. Loại bỏ số trang rác và số trang dính liền vào đầu dòng nội dung.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
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

    # Pattern dòng chỉ chứa độc lập số trang
    page_standalone_pattern = re.compile(
        r"^\s*(trang|page)?\s*[\-\–\—]?\s*\d+(\s*[\/\-]\s*\d+)?\s*[\-\–\—]?\s*$",
        re.IGNORECASE
    )
    # Pattern số trang bị dính vào đầu dòng nội dung (ví dụ: '15 Điều 1. ...')
    embedded_header_pattern = re.compile(r"^\s*\d+\s+([A-ZĐƯÊÔÀÁẢÃẠa-zđưêôàáảãạ0-9].*)$")

    for idx, line in enumerate(lines):
        line_str = line.strip()
        is_edge_line = (idx < 2) or (idx >= len(lines) - 2)

        if is_edge_line:
            # 1. Xóa dòng chỉ chứa số trang
            if page_standalone_pattern.match(line_str):
                continue
            # 2. Xóa con số đứng dính ở đầu dòng nếu có nội dung chính phía sau
            match = embedded_header_pattern.match(line)
            if match:
                cleaned_lines.append(match.group(1))
                continue

        cleaned_lines.append(line)

    return "\n".join(cleaned_lines).strip()


# --------------------------------------------------------------------------
# 3d. KHÔI PHỤC KÝ TỰ "ư" BỊ RỚT (lỗi glyph trong PDF gốc)
# --------------------------------------------------------------------------

# Bằng chứng thực đo trên controlled test TCVN3:
#   - Ban đầu tưởng chỉ xảy ra sau "đ" (đường, được) -> vá hẹp theo "đ" là SAI/THIẾU,
#     vì cùng lỗi này còn xảy ra sau "ng" (người), "ph" (phương), "tr" (trường),
#     "l" (lượng), "n" (nước)... nhiều phụ âm khác nhau.
#   - Đo width ký tự dấu cách: dấu cách "nghi vấn" (3-4pt) KHÔNG khác biệt rõ với dấu
#     cách thật (3pt trung bình) -> không thể phân biệt bằng tọa độ/kích thước.
#   - Quy luật đúng tìm được: mọi chuỗi ký tự đứng ngay trước dấu cách "nghi vấn" đều
#     là 1 PHỤ ÂM ĐẦU HỢP LỆ của tiếng Việt (đ, ng, ph, tr, l, n...) và KHÔNG chứa
#     nguyên âm. Tiếng Việt không có từ nào chỉ gồm toàn phụ âm (mọi từ đều phải có
#     nguyên âm) -> nếu gặp đúng pattern "phụ âm đầu hợp lệ" + dấu cách + tiếp tục bằng
#     chữ cái thường, chắc chắn 100% đó không phải ranh giới 2 từ thật, mà là chỗ trống
#     do glyph "ư" bị lỗi rớt trong font gốc của file test này -> chèn "ư" vào đúng chỗ đó.
# Quy tắc này tổng quát cho MỌI phụ âm đầu hợp lệ, không chỉ những cái đã quan sát thấy.
_VIETNAMESE_ONSETS = [
    # Sắp xếp dài trước để tránh khớp nhầm ("ngh" phải thử trước "ng" trước "n").
    "ngh", "kh", "ph", "th", "tr", "ch", "nh", "gi", "gh", "ng", "qu",
    "b", "c", "d", "đ", "g", "h", "k", "l", "m", "n", "p", "r", "s", "t", "v", "x",
]
_VIETNAMESE_VOWELS_LOWER = (
    "aàáảãạăằắẳẵặâầấẩẫậeèéẻẽẹêềếểễệiìíỉĩịoòóỏõọôồốổỗộơờớởỡợ"
    "uùúủũụưừứửữựyỳýỷỹỵ"
)
_missing_u_horn_pattern = re.compile(
    r"(?<![A-Za-zÀ-ỹ])(" + "|".join(_VIETNAMESE_ONSETS) + r")[ \t]+"
    r"(?=[" + _VIETNAMESE_VOWELS_LOWER + r"])",
    re.IGNORECASE,
)


def fix_missing_u_horn(text: str) -> str:
    """
    Chèn lại 'ư' vào các chỗ bị rớt glyph trong PDF gốc: 'ng êi' -> 'người',
    'ph ơng' -> 'phương', 'đ êng' -> 'đường'... (xem giải thích ở trên).
    """
    if not text:
        return text

    def _replace(match: re.Match) -> str:
        return match.group(1) + "ư"

    return _missing_u_horn_pattern.sub(_replace, text)


# --------------------------------------------------------------------------
# 3a. SUY LUẬN BẢNG MÃ CŨ -> UNICODE
# --------------------------------------------------------------------------

TCVN3_MAP: Dict[str, str] = {
    # Nguyên âm thường
    "µ": "à", "¸": "á", "¶": "ả", "·": "ã", "¹": "ạ",
    "¨": "ă", "»": "ằ", "¼": "ẳ", "½": "ẵ", "¾": "ắ", "Æ": "ặ",
    "©": "â", "Ê": "ầ", "É": "ẩ", "È": "ẫ", "Ç": "ấ", "Ë": "ậ",
    "Ì": "è", "Î": "ẻ", "Ü": "ĩ", "Ð": "é", "Ö": "ệ",
    "ª": "ê", "Ó": "ề", "Ô": "ễ", "Õ": "ế",
    "«": "ô", "å": "ồ", "æ": "ổ", "ç": "ỗ", "è": "ố", "é": "ộ",
    "¬": "ơ", "ê": "ờ", "ë": "ở", "ì": "ỡ", "í": "ớ", "î": "ợ",
    "ï": "ù", "ñ": "ủ", "ü": "ũ", "ó": "ú", "ô": "ụ",
    "­": "ư", "ð": "ừ", "÷": "ữ", "ø": "ứ", "ù": "ự", "ú": "ứ",
    "Þ": "ị", "Ò": "ề", "Ø": "ỉ", "Ý": "í", "×": "ì",
    # Bản đồ này được hiệu chỉnh từ controlled test TCVN3 và cần đối chiếu lại nếu
    # gặp corpus mới hoặc nguồn font khác.
    "ß": "ò", "ö": "ử", "õ": "õ", "ã": "ó", "ä": "ọ",
    "®": "đ", "§": "Đ",
}

# Bản đồ VNI cũng được hiệu chỉnh từ controlled test Unicode -> UniKey -> PDF,
# không nên coi là một bảng tra cứu hoàn chỉnh cho mọi nguồn VNI khác nhau.
VNI_MAP: Dict[str, str] = {
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
    "æ": "ỉ",
    "ñ": "đ", "Ñ": "Đ", "ò": "ị", "Û": "Ủ", "UÛ": "Ủ",
    "đ": "đ", "Đ": "Đ",
}

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
    Wordlist tiếng Việt mở rộng ~300 từ thông dụng.

    Ghi chú (Tuần 2 -> hậu kiểm): confidence đo được trên corpus thật (0.22-0.57) cho
    thấy wordlist 150 từ cũ (chỉ gồm từ vựng hành chính/pháp lý) match tỉ lệ thấp với
    các thể loại văn bản khác (báo cáo doanh nghiệp, giáo trình lịch sử...) dù bảng mã
    được nhận diện đúng. Bổ sung một lớp từ chức năng/hư từ tần suất cao (và, của, là,
    trong, để, khi, với, nếu, thì, mà...) — nhóm từ này xuất hiện dày đặc trong MỌI văn
    bản tiếng Việt bất kể chủ đề, nên giúp tăng độ ổn định của chỉ số confidence và giảm
    rủi ro rơi vào trạng thái `ambiguous` oan khi 2 điểm số candidate gần nhau.
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
    return function_words | domain_words


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
    uppercase_markers = {"§", "®", "Ñ", "Þ"}
    uppercase_count = sum(
        1 for ch in raw_token
        if (ch.isascii() and ch.isalpha() and ch.isupper()) or ch in uppercase_markers
    )
    lowercase_count = sum(1 for ch in raw_token if ch.isascii() and ch.isalpha() and ch.islower())
    return uppercase_count >= 2 and lowercase_count == 0


def _restore_all_caps_words(raw_text: str, normalized_text: str) -> str:
    """
    Khôi phục chữ HOA cho các token vốn là ALL-CAPS trong raw text.
    Bước này chỉ dùng để đưa output Unicode về đúng hình thức, không tham gia quyết định
    chọn bảng mã.
    """
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
    """
    Ánh xạ thử theo từng bảng mã ứng viên bằng một lần thay thế regex.
    Phần "one-pass" chỉ áp dụng cho bước thay ký tự, không phải cho quyết định chọn
    bảng mã: quyết định cuối cùng vẫn dựa trên điểm khớp từ điển tiếng Việt.
    """
    if not text or not char_map:
        return text

    sorted_keys = sorted(char_map.keys(), key=len, reverse=True)
    pattern = re.compile("|".join(re.escape(k) for k in sorted_keys))

    def replace_match(match: re.Match) -> str:
        return char_map[match.group(0)]

    return pattern.sub(replace_match, text)


def _score_candidate_text(raw_text: str, text: str, wordlist: Set[str]) -> Tuple[str, float]:
    """
    Chuẩn hóa và chấm điểm ứng viên sau khi sửa các lỗi glyph có thể ảnh hưởng trực tiếp
    đến tỉ lệ khớp từ điển. Việc chèn lại 'ư' giúp confidence phản ánh đúng hơn chất lượng
    văn bản sau chuẩn hóa, đặc biệt trên TCVN3.
    """
    normalized = unicodedata.normalize("NFC", fix_missing_u_horn(text))
    normalized = _restore_all_caps_words(raw_text, normalized)
    dict_score = _dict_match_ratio(normalized, wordlist)

    words = re.findall(r"[^\W\d_]+", normalized, flags=re.UNICODE)
    accent_score = _accent_coverage_ratio(normalized)

    if dict_score > 0.0:
        if len(words) <= 2:
            return normalized, dict_score + (accent_score * 0.15)
        return normalized, dict_score

    if len(words) <= 2:
        # Fallback cho chuỗi ngắn: nếu candidate đã phục hồi được ký tự có dấu tiếng Việt,
        # giữ một điểm số nhỏ để phân biệt với candidate Unicode gốc không đổi.
        return normalized, accent_score * 0.15

    return normalized, 0.0


def normalize_encoding(
    text: str,
    wordlist: Optional[Set[str]] = None,
    ambiguous_margin: float = 0.05,
) -> EncodingDetectionResult:
    wordlist = wordlist or default_wordlist()
    candidates: List[Tuple[Optional[str], str, float]] = []

    nfc_original = unicodedata.normalize("NFC", text)
    nfc_original, original_score = _score_candidate_text(text, nfc_original, wordlist)
    candidates.append((None, nfc_original, original_score))

    for enc_name, char_map in ENCODING_MAPS.items():
        converted = _try_decode_with_map(text, char_map)
        converted, score = _score_candidate_text(text, converted, wordlist)
        if converted == nfc_original:
            continue
        candidates.append((enc_name, converted, score))

    candidates.sort(key=lambda c: c[2], reverse=True)
    best_enc, best_text, best_score = candidates[0]
    second_score = candidates[1][2] if len(candidates) > 1 else 0.0

    # Nếu hai ứng viên sát nhau, không nên khẳng định chắc chắn một bảng mã duy nhất.
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


def normalize_page_text(raw_text: str, wordlist: Optional[Set[str]] = None) -> NormalizationResult:
    wordlist = wordlist or default_wordlist()

    # 1. Cắt bỏ số trang rác / Header / Footer dính đầu dòng
    text_no_artifacts = remove_page_number_artifacts(raw_text)

    # 2. Thử các ánh xạ bảng mã cũ và chọn ứng viên tốt nhất theo điểm khớp từ điển
    enc_result = normalize_encoding(text_no_artifacts, wordlist=wordlist)

    # 2.5. Khôi phục ký tự "ư" bị rớt do lỗi glyph trong PDF gốc
    text_fixed_spacing = fix_missing_u_horn(enc_result.normalized_text)

    # 3. Phát hiện mất dấu
    diac_result = detect_missing_diacritics(text_fixed_spacing, wordlist=wordlist)

    return NormalizationResult(
        normalized_text=text_fixed_spacing,
        detected_encoding=enc_result.detected_encoding,
        encoding_confidence=enc_result.confidence,
        encoding_warning=enc_result.warning,
        likely_missing_diacritics=diac_result.likely_missing_diacritics,
    )


if __name__ == "__main__":
    sample_normal = "Hiến pháp nước Cộng hòa xã hội chủ nghĩa Việt Nam quy định các quyền cơ bản."
    sample_nodiacritic = "Hien phap nuoc Cong hoa xa hoi chu nghia Viet Nam quy dinh cac quyen co ban."

    for label, sample in [("Chuẩn (có dấu)", sample_normal), ("Mất dấu", sample_nodiacritic)]:
        res = normalize_page_text(sample)
        print(f"--- {label} ---")
        print(f"  encoding phát hiện: {res.detected_encoding} (confidence={res.encoding_confidence:.2f})")
        print(f"  mất dấu?: {res.likely_missing_diacritics}")
"""
script/app_ui_style.py — Lớp trình bày (CSS + mảnh HTML tĩnh) của giao diện
"Chat with Your PDF", tách riêng khỏi script/app_ui.py.

File này CHỈ chứa markup/CSS thuần túy, KHÔNG gọi bất kỳ hàm nào từ source/,
KHÔNG chứa logic pipeline hay đọc st.session_state — mục đích tách ra là để
app_ui.py chỉ còn logic pipeline + luồng UI, dễ đọc và dễ review hơn khi
báo cáo (Việc A, Tuần 6).

Đổi CSS ở đây KHÔNG ảnh hưởng tới bất kỳ hành vi RAG nào.
"""
from __future__ import annotations

import streamlit as st

APP_CSS = """
<style>
/* =========================================================
   APP SHELL
   ========================================================= */
.stApp {
    background:
        radial-gradient(900px 500px at 85% 5%,
            rgba(124, 58, 237, .055), transparent 60%),
        radial-gradient(800px 500px at 5% 20%,
            rgba(59, 130, 246, .045), transparent 60%);
}

.block-container {
    max-width: 1120px;
    padding-top: 4.5rem !important;
    padding-bottom: 7rem !important;
}

#MainMenu { visibility: hidden; }
footer { visibility: hidden; }

/* =========================================================
   TOP BRAND
   ========================================================= */
.topbar {
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 20px;
    margin-bottom: 24px;
}

.brand {
    display: flex;
    align-items: center;
    gap: 12px;
}

.brand-mark {
    width: 42px;
    height: 42px;
    border-radius: 13px;
    display: flex;
    align-items: center;
    justify-content: center;
    background: linear-gradient(135deg,
        rgba(99,102,241,.18),
        rgba(139,92,246,.12));
    border: 1px solid rgba(99,102,241,.13);
    font-size: 21px;
}

.brand-name {
    font-size: 1.22rem;
    font-weight: 780;
    letter-spacing: -.04em;
    line-height: 1.1;
}

.brand-desc {
    margin-top: 3px;
    color: rgba(128,128,128,.88);
    font-size: .79rem;
}

.brand-pill {
    padding: 7px 11px;
    border: 1px solid rgba(128,128,128,.14);
    border-radius: 999px;
    color: rgba(128,128,128,.9);
    background: rgba(255,255,255,.035);
    font-size: .72rem;
    white-space: nowrap;
}

/* =========================================================
   UPLOAD LANDING
   ========================================================= */
.upload-shell {
    max-width: 850px;
    margin: 40px auto 0;
    text-align: center;
}

.upload-eyebrow {
    display: inline-block;
    margin-bottom: 15px;
    padding: 6px 10px;
    border: 1px solid rgba(99,102,241,.13);
    border-radius: 999px;
    background: rgba(99,102,241,.055);
    color: rgba(79,70,229,.95);
    font-size: .70rem;
    font-weight: 750;
    letter-spacing: .08em;
    text-transform: uppercase;
}

.upload-title {
    margin: 0;
    font-size: clamp(2rem, 4vw, 3.1rem);
    font-weight: 800;
    letter-spacing: -.065em;
    line-height: 1.03;
}

.upload-subtitle {
    max-width: 650px;
    margin: 15px auto 28px;
    color: rgba(128,128,128,.95);
    font-size: .96rem;
    line-height: 1.6;
}

.upload-panel {
    padding: 8px;
    border: 1px solid rgba(128,128,128,.15);
    border-radius: 22px;
    background: rgba(255,255,255,.45);
    box-shadow:
        0 22px 70px rgba(0,0,0,.045),
        inset 0 1px 0 rgba(255,255,255,.65);
}

[data-testid="stFileUploader"] {
    margin: 0 !important;
}

[data-testid="stFileUploaderDropzone"] {
    min-height: 145px;
    border-radius: 17px !important;
    border: 1px dashed rgba(99,102,241,.28) !important;
    background: rgba(99,102,241,.025) !important;
}

.upload-hint {
    margin-top: 12px;
    color: rgba(128,128,128,.75);
    font-size: .74rem;
}

/* =========================================================
   ACTIVE DOCUMENT
   ========================================================= */
.active-doc {
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 15px;
    margin: 24px 0 28px;
    padding: 13px 15px;
    border: 1px solid rgba(128,128,128,.14);
    border-radius: 17px;
    background: rgba(255,255,255,.38);
    box-shadow: 0 8px 28px rgba(0,0,0,.025);
}

.active-doc-left {
    display: flex;
    align-items: center;
    gap: 11px;
    min-width: 0;
}

.active-doc-icon {
    width: 39px;
    height: 39px;
    flex: 0 0 auto;
    display: flex;
    align-items: center;
    justify-content: center;
    border-radius: 11px;
    background: rgba(99,102,241,.09);
}

.active-doc-name {
    font-size: .86rem;
    font-weight: 700;
    overflow-wrap: anywhere;
}

.active-doc-ready {
    margin-top: 2px;
    color: #16a34a;
    font-size: .72rem;
    font-weight: 600;
}

/* =========================================================
   WELCOME / STARTERS
   ========================================================= */
.welcome {
    margin: 42px auto 25px;
    text-align: center;
}

.welcome-icon {
    font-size: 31px;
    margin-bottom: 8px;
}

.welcome-title {
    font-size: 1.30rem;
    font-weight: 760;
    letter-spacing: -.035em;
}

.welcome-text {
    max-width: 650px;
    margin: 7px auto 0;
    color: rgba(128,128,128,.88);
    font-size: .86rem;
    line-height: 1.5;
}

.starter-label {
    margin: 0 0 10px;
    color: rgba(128,128,128,.72);
    font-size: .68rem;
    font-weight: 780;
    letter-spacing: .11em;
    text-transform: uppercase;
}

.starter {
    min-height: 70px;
    padding: 13px 14px;
    border: 1px solid rgba(128,128,128,.13);
    border-radius: 16px;
    background: rgba(255,255,255,.28);
}

.starter-icon {
    margin-bottom: 5px;
    font-size: 16px;
}

.starter-text {
    color: rgba(30,30,30,.90);
    font-size: .79rem;
    line-height: 1.38;
}

/* =========================================================
   SOURCE CHIPS
   ========================================================= */
.source-label {
    margin-top: 12px;
    margin-bottom: 4px;
    color: rgba(128,128,128,.86);
    font-size: .72rem;
    font-weight: 700;
}

.source-chip {
    display: inline-block;
    margin: 3px 5px 2px 0;
    padding: 5px 9px;
    border: 1px solid rgba(99,102,241,.12);
    border-radius: 999px;
    background: rgba(99,102,241,.045);
    font-size: .72rem;
}

/* =========================================================
   SIDEBAR
   ========================================================= */
section[data-testid="stSidebar"] {
    border-right: 1px solid rgba(128,128,128,.10);
}

.sidebar-brand {
    padding-top: 7px;
    margin-bottom: 20px;
}

.sidebar-title {
    font-size: 1.02rem;
    font-weight: 780;
    letter-spacing: -.025em;
}

.sidebar-subtitle {
    margin-top: 4px;
    color: rgba(128,128,128,.84);
    font-size: .76rem;
    line-height: 1.45;
}

.sidebar-file {
    padding: 12px;
    border: 1px solid rgba(128,128,128,.13);
    border-radius: 14px;
    background: rgba(255,255,255,.20);
    font-size: .78rem;
    line-height: 1.45;
}

.sidebar-file-name {
    font-weight: 700;
    overflow-wrap: anywhere;
}

.sidebar-ready {
    color: #16a34a;
    font-size: .70rem;
}

.stButton > button {
    border-radius: 11px;
    font-weight: 650;
}

/* =========================================================
   CHAT
   ========================================================= */
[data-testid="stChatMessage"] {
    border-radius: 18px;
}

[data-testid="stChatInput"] {
    border-radius: 17px;
}

/* =========================================================
   MOBILE
   ========================================================= */
@media (max-width: 700px) {
    .block-container {
        padding-top: 4rem !important;
        padding-left: 1rem;
        padding-right: 1rem;
    }

    .brand-pill {
        display: none;
    }

    .upload-shell {
        margin-top: 28px;
    }

    .active-doc {
        align-items: flex-start;
    }
}
</style>
"""


def inject_css() -> None:
    """Chèn CSS toàn cục vào trang. Gọi 1 lần duy nhất, ngay sau set_page_config."""
    st.markdown(APP_CSS, unsafe_allow_html=True)

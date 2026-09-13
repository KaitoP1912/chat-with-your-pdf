@echo off
REM chay_ung_dung.bat - Double-click file nay de mo giao dien Chat with Your PDF.
REM Khong can nho lenh streamlit dai, khong bi loi duong dan venv.

cd /d "%~dp0"
call venv\Scripts\activate.bat
python -m streamlit run script\app_ui.py --server.fileWatcherType none
pause
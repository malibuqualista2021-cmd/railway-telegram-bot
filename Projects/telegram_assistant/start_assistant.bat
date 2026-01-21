@echo off
REM ==========================================
REM Telegram Asistan v4.2 - Windows Startup Script
REM ==========================================

SETLOCAL

REM Yapılandırma
SET OLLAMA_PATH=C:\Users\malib\AppData\Local\Programs\Ollama
SET BOT_PATH=C:\Users\malib
SET BOT_SCRIPT=telegram_asistant_v42.py
SET LOG_FILE=%BOT_PATH%\bot_startup.log

REM Log dosyasına yaz
echo [%DATE% %TIME%] Asistan başlatılıyor... >> "%LOG_FILE%"

REM Ollama başlat (zaten çalışıyorsa atla)
tasklist /FI "IMAGENAME eq ollama.exe" 2>NUL | find /I /N "ollama.exe">NUL
IF "%ERRORLEVEL%"=="0" (
    echo [%DATE% %TIME%] Ollama zaten çalışıyor >> "%LOG_FILE%"
) ELSE (
    echo [%DATE% %TIME%] Ollama başlatılıyor... >> "%LOG_FILE%"
    start /B "" "%OLLAMA_PATH%\ollama.exe" serve >> "%LOG_FILE%" 2>&1
    REM Ollama'nın başlaması için bekle
    timeout /t 5 /nobreak > nul
)

REM Python kontrolü
python --version >> "%LOG_FILE%" 2>&1
IF %ERRORLEVEL% NEQ 0 (
    echo [%DATE% %TIME%] HATA: Python bulunamadı! >> "%LOG_FILE%"
    pause
    exit /b 1
)

REM Botu başlat
cd /D "%BOT_PATH%"
echo [%DATE% %TIME%] Bot başlatılıyor... >> "%LOG_FILE%"
python "%BOT_SCRIPT%" >> "%LOG_FILE%" 2>&1

ENDLOCAL

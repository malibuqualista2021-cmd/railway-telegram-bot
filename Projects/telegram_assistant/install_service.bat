@echo off
REM ==========================================
REM NSSM ile Windows Servisi Kurulumu
REM ==========================================

SETLOCAL ENABLEDELAYEDEXPANSION

SET SERVICE_NAME=TelegramAsistan
SET OLLAMA_PATH=C:\Users\malib\AppData\Local\Programs\Ollama
SET BOT_PATH=C:\Users\malib
SET BOT_SCRIPT=telegram_asistant_v42.py

echo ==========================================
echo   Telegram Asistan - Servis Kurulumu
echo ==========================================
echo.

REM NSSM kontrolü
where nssm >nul 2>&1
IF %ERRORLEVEL% NEQ 0 (
    echo [!] NSSM bulunamadı!
    echo.
    echo NSSM'yi indirin:
    echo https://nssm.cc/download
    echo.
    echo İndirdikten sonra bu script'i tekrar çalıştırın.
    pause
    exit /b 1
)

echo [+] NSSM bulundu
echo.

REM Servisi kaldır (varsa)
sc query "%SERVICE_NAME%" >nul 2>&1
IF %ERRORLEVEL% EQU 0 (
    echo [*] Mevcut servis kaldırılıyor...
    nssm stop "%SERVICE_NAME%"
    nssm remove "%SERVICE_NAME%" confirm
    timeout /t 2 /nobreak > nul
)

REM Ollama servisi
echo [+] Ollama servisi kuruluyor...
nssm install Ollama "%OLLAMA_PATH%\ollama.exe" serve
nssm set Ollama AppDirectory "%OLLAMA_PATH%"
nssm set Ollama DisplayName "Ollama LLM Server"
nssm set Ollama Description "Yerel LLM sunucusu - GLM 4"
nssm set Ollama Start SERVICE_AUTO_START
nssm set Ollama AppRestartDelay 5000
echo [+] Ollama servisi kuruldu
echo.

REM Bot servisi
echo [+] Telegram Asistan servisi kuruluyor...

REM Geçici wrapper script oluştur
set WRAPPER=%BOT_PATH%\service_wrapper.bat
echo @echo off > "%WRAPPER%"
echo SETLOCAL >> "%WRAPPER%"
echo cd /D "%BOT_PATH%" >> "%WRAPPER%"
echo python "%BOT_SCRIPT%" >> "%WRAPPER%"
echo ENDLOCAL >> "%WRAPPER%"

nssm install "%SERVICE_NAME}" "%WRAPPER%"
nssm set "%SERVICE_NAME%" AppDirectory "%BOT_PATH%"
nssm set "%SERVICE_NAME%" DisplayName "Telegram AI Asistan v4.2"
nssm set "%SERVICE_NAME%" Description "GLM 4 tabanlı Telegram asistan - Kalıcı hafıza sistemi"
nssm set "%SERVICE_NAME%" Start SERVICE_AUTO_START
nssm set "%SERVICE_NAME%" AppRestartDelay 10000
nssm set "%SERVICE_NAME%" AppRestartSeconds 60

REM Servis bağımlılığı (Ollama'dan önce başlamasın)
nssm set "%SERVICE_NAME%" AppDependsOn Ollama

REM Çıktı yönlendirme (opsiyonel)
nssm set "%SERVICE_NAME%" AppStdout "%BOT_PATH%\service.log"
nssm set "%SERVICE_NAME%" AppStderr "%BOT_PATH%\service_error.log"

echo [+] Telegram Asistan servisi kuruldu
echo.

echo ==========================================
echo   Kurulum Tamamlandı!
echo ==========================================
echo.
echo Komutlar:
echo   nssm start Ollama           -- Ollama'yı başlat
echo   nssm start %SERVICE_NAME%   -- Bot'u başlat
echo   nssm stop Ollama            -- Ollama'yı durdur
echo   nssm stop %SERVICE_NAME%    -- Bot'u durdur
echo.
echo Servisleri başlatmak için bir tuşa basin...
pause > nul

nssm start Ollama
timeout /t 5 /nobreak > nul
nssm start "%SERVICE_NAME%"

echo.
echo [*] Servisler başlatıldı!
echo.
echo Durum kontrolü:
nssm status Ollama
nssm status "%SERVICE_NAME%"

echo.
pause

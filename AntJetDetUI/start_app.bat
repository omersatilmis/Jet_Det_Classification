@echo off
echo ===================================================
echo   ANTJETDET UI - BASLATMA PROTOKOLU
echo ===================================================
echo.
echo [1/3] Arka planda acik kalmis eski "hayalet" sunucular temizleniyor...
taskkill /F /IM node.exe /T >nul 2>&1
taskkill /F /IM python.exe /T >nul 2>&1
timeout /t 2 /nobreak >nul

echo [2/3] AI Backend (FastAPI + MMDetection) baslatiliyor...
start "AntJetDet Backend" cmd /k "cd /d %~dp0backend && call %USERPROFILE%\miniconda3\Scripts\activate.bat jetdet && python main.py"

echo Backend modeli yuklerken 15 saniye bekleniyor...
timeout /t 15 /nobreak >nul

echo [3/3] Frontend (Kullanici Arayuzu) tarayicida aciliyor...
call npm run dev -- --open

echo.
echo Sistem kapatiliyor, tum arka plan islemleri temizleniyor...
taskkill /F /IM node.exe /T >nul 2>&1
taskkill /F /IM python.exe /T >nul 2>&1
exit

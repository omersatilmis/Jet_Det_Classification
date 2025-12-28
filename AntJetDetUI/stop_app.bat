@echo off
echo ===================================================
echo   ANTJETDET UI - ACIL DURUM KAPATMA (KILL SWITCH)
echo ===================================================
echo.
echo Arka planda calisan tum AI Backend (Python) ve Frontend (Node) islemleri zorla kapatiliyor...
echo.

taskkill /F /IM node.exe /T 
taskkill /F /IM python.exe /T 

echo.
echo ===================================================
echo   TEMIZLIK TAMAMLANDI. SISTEM SIFIRLANDI.
echo ===================================================
pause

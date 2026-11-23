@echo off
title Stop Freight Twin
powershell -NoProfile -ExecutionPolicy Bypass -File "%USERPROFILE%\Desktop\backend\scripts\stop_public.ps1"
echo.
pause

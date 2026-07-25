@echo off
chcp 65001 >nul
title SACR Tool Installer

echo ========================================================
echo   SACR Tool — One-Click Installer
echo ========================================================
echo.

REM Check Python
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo [ERROR] Python not found. Install Python from python.org first.
    pause
    exit /b 1
)

REM Install the package
echo [1/3] Installing SACR Tool...
pip install git+https://github.com/wepexrm-bot/SACR-Tool.git
if %errorlevel% neq 0 (
    echo [ERROR] Installation failed.
    pause
    exit /b 1
)

REM Find Python Scripts folder
echo [2/3] Detecting Python Scripts folder...
for /f "delims=" %%i in ('python -c "import sys; print(sys.base_exec_prefix)"') do set "BASE=%%i"
set "SCRIPTS_DIR=%BASE%\Scripts"

if not exist "%SCRIPTS_DIR%" (
    for /f "delims=" %%i in ('python -c "import site; print(site.USER_BASE)"') do set "SCRIPTS_DIR=%%i\Scripts"
)

REM Add to PATH (user level, no admin needed)
echo [3/3] Adding SACR Tool to PATH...
setx PATH "%PATH%;%SCRIPTS_DIR%" >nul

REM Add to current session too so it works immediately
set "PATH=%PATH%;%SCRIPTS_DIR%"

echo.
echo ========================================================
echo   INSTALLATION COMPLETE!
echo ========================================================
echo.
echo   Try it now:  sacr_cli --version
echo   (Close and reopen terminal for permanent effect)
echo.
pause
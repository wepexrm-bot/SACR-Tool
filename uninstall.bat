@echo off
chcp 65001 >nul
title SACR Tool Uninstaller

echo ========================================================
echo   SACR Tool — Uninstaller
echo ========================================================
echo.

REM Uninstall via pip
pip uninstall sacr-tool -y >nul 2>&1

REM Remove from PATH
for /f "delims=" %%i in ('python -c "import sys; import site; p = site.USER_BASE+'\\Scripts' if hasattr(site,'USER_BASE') else sys.base_exec_prefix+'\\Scripts'; print(p)" 2>nul') do set "SCRIPTS_DIR=%%i"
if not defined SCRIPTS_DIR set "SCRIPTS_DIR=%APPDATA%\Python\Python310\Scripts"

echo [INFO] Removing %SCRIPTS_DIR% from PATH...
setlocal enabledelayedexpansion
set "NEWPATH="
for %%a in ("%PATH:;=";"%") do (
    if /i "%%~a" neq "%SCRIPTS_DIR%" (
        if defined NEWPATH set "NEWPATH=!NEWPATH!;"
        set "NEWPATH=!NEWPATH!%%~a"
    )
)
endlocal & set "PATH=%NEWPATH%"
setx PATH "%NEWPATH%" >nul

echo.
echo ========================================================
echo   UNINSTALL COMPLETE!
echo ========================================================
echo.
echo   Close and reopen your terminal for changes to take effect.
echo.
pause
@echo off
:: Runs DeepLabCut tracking through behavython-cli with timing, progress and an ETA. Runs analyze_dlc_videos.py from this folder.
::
:: Usage: drag one or more folders and/or videos onto this file (it asks for the network), or
::   analyze_dlc_videos.bat <folders or videos> [--network cd1^|c57] [--dlc-config PATH] [--recursive] [--dry-run] [--rate FPS]
::
:: Needs the miniforge env %USERPROFILE%\miniforge3\envs\behavython with DeepLabCut, and this file inside the behavython repo's scripts folder.
setlocal
set "PY=%USERPROFILE%\miniforge3\envs\behavython\python.exe"
if not exist "%PY%" (
    echo [ERROR] Python not found: %PY%
    echo Create the behavython miniforge env with DeepLabCut, then run this again.
    pause
    exit /b 1
)
if "%~1"=="" (
    "%PY%" "%~dp0analyze_dlc_videos.py" --help
    pause
    exit /b 1
)
"%PY%" "%~dp0analyze_dlc_videos.py" %*
set "CODE=%ERRORLEVEL%"
pause
exit /b %CODE%

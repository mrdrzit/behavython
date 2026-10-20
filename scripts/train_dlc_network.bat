@echo off
:: Trains a DeepLabCut PyTorch network with a live progress view. Runs train_dlc_network.py from this folder.
::
:: Usage: drag a network folder onto this file, or
::   train_dlc_network.bat <network folder or config.yaml> [--watch] [--overwrite] [--poll SECONDS]
::
:: Needs the miniforge env %USERPROFILE%\miniforge3\envs\behavython with DeepLabCut 3.0.2 (PyTorch) and PyYAML.
:: The script does not import behavython, so the repo location does not matter, only that the .py sits next to this file.
setlocal
set "PY=%USERPROFILE%\miniforge3\envs\behavython\python.exe"
if not exist "%PY%" (
    echo [ERROR] Python not found: %PY%
    echo Create the behavython miniforge env with DeepLabCut 3.0.2 for PyTorch and PyYAML, then run this again.
    pause
    exit /b 1
)
if "%~1"=="" (
    "%PY%" "%~dp0train_dlc_network.py" --help
    pause
    exit /b 1
)
"%PY%" "%~dp0train_dlc_network.py" %*
set "CODE=%ERRORLEVEL%"
pause
exit /b %CODE%

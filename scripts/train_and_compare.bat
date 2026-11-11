@echo off
:: Trains a new DeepLabCut network, then compares it with other networks on a folder of videos.
:: Runs train_dlc_network.py, then compare_networks.py, from this folder, in the behavython miniforge env.
::
:: Usage:
::   train_and_compare.bat <new project folder> <eval folder> <config.yaml of each network to compare with> [...]
:: The new network is compared too; it does not need to be listed again.
setlocal
:: shift also moves %0, so the script folder is saved before any shift
set "HERE=%~dp0"
set "PY=%USERPROFILE%\miniforge3\envs\behavython\python.exe"
if not exist "%PY%" (
    echo [ERROR] Python not found: %PY%
    pause
    exit /b 1
)
if "%~3"=="" (
    echo Usage: train_and_compare.bat ^<new project folder^> ^<eval folder^> ^<config.yaml^> [...]
    pause
    exit /b 1
)
set "NEW=%~1"
set "EVAL=%~2"
shift
shift
set "OTHERS="
:collect
if "%~1"=="" goto run
set OTHERS=%OTHERS% "%~1"
shift
goto collect
:run
echo === Training %NEW%  (started %date% %time%)
"%PY%" "%HERE%train_dlc_network.py" "%NEW%"
echo === Comparing  (started %date% %time%)
"%PY%" "%HERE%compare_networks.py" "%EVAL%" %OTHERS% "%NEW%"
echo === Finished %date% %time%
pause

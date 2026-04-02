@echo off
setlocal

set "ENV_NAME=behavython"
set "APP_MODULE=src.behavython.main"
set "CONDA_CMD="

echo.
echo [Behavython Launcher]
echo.

:: --- Find conda/mamba ---
echo [INFO] Searching for conda/mamba...

for %%D in (
    "%USERPROFILE%\miniforge3"
    "%USERPROFILE%\mambaforge"
    "%USERPROFILE%\mambaforge3"
    "%USERPROFILE%\miniconda3"
    "%ProgramData%\miniforge3"
    "%ProgramData%\mambaforge"
    "%ProgramData%\mambaforge3"
    "%ProgramData%\Miniconda3"
) do (
    if not defined CONDA_CMD (
        if exist "%%~D\Scripts\mamba.exe" (
            set "CONDA_CMD=%%~D\Scripts\mamba.exe"
            echo [INFO] Found mamba at %%~D
        ) else if exist "%%~D\condabin\conda.bat" (
            set "CONDA_CMD=%%~D\condabin\conda.bat"
            echo [INFO] Found conda at %%~D
        )
    )
)

:: --- If found, test env and run ---
if defined CONDA_CMD (
    echo [INFO] Checking environment "%ENV_NAME%"...

    call "%CONDA_CMD%" run -n "%ENV_NAME%" python -c "import sys" >nul 2>&1

    if not errorlevel 1 (
        echo [INFO] Environment found. Launching Behavython...
        call "%CONDA_CMD%" run --no-capture-output -n "%ENV_NAME%" python -m %APP_MODULE%
        goto :end
    ) else (
        echo [WARN] Environment "%ENV_NAME%" not found or invalid.
    )
) else (
    echo [WARN] No conda/mamba installation found.
	echo [WARN] Please follow the repo instructions to install Behavython on your computer
)

:: --- Fallback ---
echo [INFO] Falling back to system Python...
python -m %APP_MODULE%

:noErrorExit
echo.
if %ERRORLEVEL% neq 0 (
    goto :errorExit
) else (
    goto :end
)
echo.
pause
endlocal

:errorExit
echo.
echo [ERROR] An unknown error occurred.
echo [ERROR] Please check your Python installation and environment setup.
pause
endlocal

:end
echo.
echo [INFO] Behavython has exited. Thank you for using it!
pause
endlocal
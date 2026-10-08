@echo off
setlocal

set "ENV_NAME=behavython"
set "APP_MODULE=behavython.main"
set "CONDA_CMD="

echo.
echo [Behavython Launcher]
echo.

:: --- Find conda/mamba ---
echo [INFO] Searching for conda/mamba...

:: Check PATH first
for /f "delims=" %%I in ('where mamba.bat 2^>nul') do if not defined CONDA_CMD set "CONDA_CMD=%%~fI"
if not defined CONDA_CMD for /f "delims=" %%I in ('where mamba.exe 2^>nul') do if not defined CONDA_CMD set "CONDA_CMD=%%~fI"
if not defined CONDA_CMD for /f "delims=" %%I in ('where conda.bat 2^>nul') do if not defined CONDA_CMD set "CONDA_CMD=%%~fI"
if not defined CONDA_CMD for /f "delims=" %%I in ('where conda.exe 2^>nul') do if not defined CONDA_CMD set "CONDA_CMD=%%~fI"
if defined CONDA_CMD (
    echo [INFO] Found on PATH: %CONDA_CMD%
) else (
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
        goto :errorEnvNotFound
    )
) else (
    echo [WARN] No conda/mamba installation found.
    echo [WARN] Please follow the repo instructions to install Behavython on your computer.
)
:: --- Fallback ---
echo [INFO] Falling back to system Python...
python -m %APP_MODULE%

if %ERRORLEVEL% neq 0 (
    goto :errorSystemPython
) else (
    goto :end
)

:errorEnvNotFound
cls
echo ========================================================================
echo                       BEHAVYTHON LAUNCH ERROR
echo ========================================================================
echo.
echo  [PROBLEM]
echo  The required environment "%ENV_NAME%" was not found.
echo.
echo  Behavython cannot start because it requires packages installed
echo  inside a conda/mamba environment with this specific name.
echo.
echo ========================================================================
echo  HOW TO RESOLVE THIS
echo ========================================================================
echo.
echo  1. Check your existing environments:
echo     Open a terminal (Miniforge Prompt, Anaconda Prompt, or PowerShell)
echo     and run:
echo.
echo         mamba env list
echo         (or: conda env list)
echo.
echo  2. If the environment that you want to use has a different name:
echo     Open this launcher file in Notepad:
echo.
echo         "%~f0"
echo.
echo     Change line 4 from:
echo         set "ENV_NAME=%ENV_NAME%"
echo     to:
echo         set "ENV_NAME=your_actual_env_name"
echo.
echo     Save the file and run it again.
echo.
echo  3. If you have not created an environment yet:
echo     Follow the repository installation instructions to set up the
echo     environment and required dependencies.
echo.
echo ========================================================================
echo.
pause
endlocal
exit /b 1

:errorSystemPython
cls
echo ========================================================================
echo                       BEHAVYTHON LAUNCH ERROR
echo ========================================================================
echo.
echo  [PROBLEM]
echo  Failed to launch Behavython using system Python.
echo.
echo  No valid conda/mamba environment was detected, and system Python
echo  could not run module "%APP_MODULE%".
echo.
echo ========================================================================
echo  HOW TO RESOLVE THIS
echo ========================================================================
echo.
echo  1. Install Miniforge or Miniconda on this system.
echo  2. Follow the repository setup instructions to create an environment.
echo  3. Ensure the environment name matches line 4 of this launcher:
echo         "%~f0"
echo.
echo ========================================================================
echo.
pause
endlocal
exit /b 1

:end
echo.
echo [INFO] Behavython has exited. Thank you for using it!
pause
endlocal
@echo off
setlocal EnableExtensions EnableDelayedExpansion

:: ======================================================
:: Behavython installer (pip-first)
:: - mamba/conda used mainly for env creation + run
:: - most packages installed with pip
:: - optional conda CUDA/cuDNN for TensorFlow on Windows
:: ======================================================

set "ENV_NAME=behavython"
set "PYTHON_VERSION=3.10.18"
set "TARGET_DIR=%USERPROFILE%\Documents\behavython"
set "REPO_URL=https://github.com/mrdrzit/behavython.git"

set "CONDA_ROOT="
set "CONDA_CMD="
set "CONDA_KIND="

echo.
echo ======================================================
echo Behavython installer
echo ======================================================
echo Environment : %ENV_NAME%
echo Python      : %PYTHON_VERSION%
echo Target dir  : %TARGET_DIR%
echo.

:: ------------------------------------------------------
:: STEP 0 - Find Miniforge / Mambaforge / Miniconda
:: ------------------------------------------------------
echo ======================================================
echo [STEP 0] Detecting conda/mamba installation
echo ======================================================

for %%D in (
    "%USERPROFILE%\miniforge3"
    "%USERPROFILE%\mambaforge3"
    "%USERPROFILE%\miniconda3"
) do (
    if not defined CONDA_CMD (
        if exist "%%~D\Scripts\mamba.exe" (
            set "CONDA_ROOT=%%~D"
            set "CONDA_CMD=%%~D\Scripts\mamba.exe"
            set "CONDA_KIND=mamba"
        ) else if exist "%%~D\condabin\conda.bat" (
            set "CONDA_ROOT=%%~D"
            set "CONDA_CMD=%%~D\condabin\conda.bat"
            set "CONDA_KIND=conda_bat"
        ) else if exist "%%~D\Scripts\conda.exe" (
            set "CONDA_ROOT=%%~D"
            set "CONDA_CMD=%%~D\Scripts\conda.exe"
            set "CONDA_KIND=conda_exe"
        )
    )
)

if not defined CONDA_CMD (
    echo [ERROR] Could not find Miniforge, Mambaforge, or Miniconda in expected locations.
    pause
    exit /b 1
)

echo [INFO] Found root : %CONDA_ROOT%
echo [INFO] Using      : %CONDA_CMD%
echo [INFO] Type       : %CONDA_KIND%
echo.

call "%CONDA_CMD%" --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] The detected conda/mamba command could not be executed.
    pause
    exit /b 1
)

:: ------------------------------------------------------
:: STEP 1 - Check Git
:: ------------------------------------------------------
echo ======================================================
echo [STEP 1] Checking Git
echo ======================================================

where git >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Git was not found in PATH.
    pause
    exit /b 1
)

echo [INFO] Git found.
echo.

:: ------------------------------------------------------
:: STEP 2 - Create environment if needed
:: ------------------------------------------------------
echo ======================================================
echo [STEP 2] Checking conda environment
echo ======================================================

call "%CONDA_CMD%" run -n %ENV_NAME% python --version >nul 2>&1
if errorlevel 1 (
    echo [INFO] Environment '%ENV_NAME%' not found. Creating...
    call "%CONDA_CMD%" create -y -n %ENV_NAME% python=%PYTHON_VERSION% pip
    if errorlevel 1 (
        echo [ERROR] Failed to create environment.
        pause
        exit /b 1
    )
) else (
    echo [INFO] Environment '%ENV_NAME%' already exists.
)

:: ------------------------------------------------------
:: STEP 3 - Upgrade pip tooling
:: ------------------------------------------------------
echo.
echo ======================================================
echo [STEP 3] Upgrading pip tooling
echo ======================================================

call "%CONDA_CMD%" run -n %ENV_NAME% python -m pip install --upgrade pip setuptools wheel
if errorlevel 1 (
    echo [ERROR] Failed to upgrade pip tooling.
    pause
    exit /b 1
)

:: ------------------------------------------------------
:: STEP 4 - Install core scientific stack with pip
:: ------------------------------------------------------
echo.
echo ======================================================
echo [STEP 4] Installing core scientific stack with pip
echo ======================================================

call "%CONDA_CMD%" run -n %ENV_NAME% python -m pip install ^
    "numpy<2.0" ^
    "matplotlib==3.8.3" ^
    seaborn ^
    pandas ^
    scipy ^
    openpyxl ^
    scikit-image ^
    tables ^
    opencv-python ^
    PyYAML ^
    tqdm ^
    debugpy ^
	flask

if errorlevel 1 (
    echo [ERROR] Core pip package installation failed.
    pause
    exit /b 1
)

:: ------------------------------------------------------
:: STEP 5 - Optional TensorFlow GPU support on Windows
:: ------------------------------------------------------
echo.
echo ======================================================
echo [STEP 5] Installing CUDA/TensorFlow stack
echo ======================================================

call "%CONDA_CMD%" install -y -n %ENV_NAME% -c conda-forge cudatoolkit=11.2 cudnn=8.1.0
if errorlevel 1 (
    echo [ERROR] Failed to install cudatoolkit/cudnn.
    pause
    exit /b 1
)

call "%CONDA_CMD%" run -n %ENV_NAME% python -m pip install "tensorflow<2.11"
if errorlevel 1 (
    echo [ERROR] TensorFlow installation failed.
    pause
    exit /b 1
)

echo.
echo [INFO] Checking TensorFlow GPU visibility...
call "%CONDA_CMD%" run -n %ENV_NAME% python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
if errorlevel 1 (
    echo [ERROR] TensorFlow GPU device check failed.
    pause
    exit /b 1
)

echo.
echo [INFO] Running TensorFlow compute test...
call "%CONDA_CMD%" run -n %ENV_NAME% python -c "import tensorflow as tf; print(tf.reduce_sum(tf.random.normal([1000, 1000])))"
if errorlevel 1 (
    echo [ERROR] TensorFlow compute test failed.
    pause
    exit /b 1
)

:: ------------------------------------------------------
:: STEP 6 - Install PyTorch with pip
:: Choose ONE of the blocks below
:: ------------------------------------------------------
echo.
echo ======================================================
echo [STEP 6] Installing PyTorch with pip
echo ======================================================

:: CPU-only:
call "%CONDA_CMD%" run -n %ENV_NAME% python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
if errorlevel 1 (
    echo [ERROR] PyTorch installation failed.
    pause
    exit /b 1
)

call "%CONDA_CMD%" run -n %ENV_NAME% python -c "import torch; print(torch.__version__)"
if errorlevel 1 (
    echo [ERROR] PyTorch import test failed.
    pause
    exit /b 1
)

call "%CONDA_CMD%" run -n %ENV_NAME% python -c "import torch; print(torch.cuda.is_available())"
if errorlevel 1 (
    echo [ERROR] PyTorch CUDA check failed.
    pause
    exit /b 1
)

:: ------------------------------------------------------
:: STEP 7 - Install DeepLabCut + extras with pip
:: ------------------------------------------------------
echo.
echo ======================================================
echo [STEP 7] Installing DeepLabCut with GUI
echo ======================================================

call "%CONDA_CMD%" run -n %ENV_NAME% python -m pip install --upgrade "deeplabcut[gui]==2.3.10" tensorpack tf_slim
if errorlevel 1 (
    echo [ERROR] DeepLabCut installation failed.
    pause
    exit /b 1
)

:: ------------------------------------------------------
:: STEP 8 - Optional PySide6 upgrade after DLC install
:: ------------------------------------------------------
echo.
echo ======================================================
echo [STEP 8] Upgrading PySide6 after DLC install
echo ======================================================

call "%CONDA_CMD%" run -n %ENV_NAME% python -m pip install --upgrade "PySide6==6.7.3"
if errorlevel 1 (
    echo [ERROR] PySide6 upgrade failed.
    pause
    exit /b 1
)

:: ------------------------------------------------------
:: STEP 9 - Verify imports
:: ------------------------------------------------------
echo.
echo ======================================================
echo [STEP 9] Verifying DeepLabCut, NumPy, PyTorch, PySide6
echo ======================================================

call "%CONDA_CMD%" run -n %ENV_NAME% python -c "import numpy; import numpy.core._multiarray_umath; print('NumPy OK:', numpy.__version__)"
if errorlevel 1 (
    echo [ERROR] NumPy binary check failed.
    pause
    exit /b 1
)

call "%CONDA_CMD%" run -n %ENV_NAME% python -c "import torch; print('Torch OK:', torch.__version__)"
if errorlevel 1 (
    echo [ERROR] Torch import test failed.
    pause
    exit /b 1
)

call "%CONDA_CMD%" run -n %ENV_NAME% python -c "from PySide6 import QtCore; print('PySide6 OK:', QtCore.__version__)"
if errorlevel 1 (
    echo [ERROR] PySide6 import test failed.
    pause
    exit /b 1
)

call "%CONDA_CMD%" run -n %ENV_NAME% python -c "import torch; import deeplabcut; print('DeepLabCut OK:', deeplabcut.__version__)"
if errorlevel 1 (
    echo [ERROR] DeepLabCut import test failed.
    pause
    exit /b 1
)

:: ------------------------------------------------------
:: STEP 10 - Clone or update repo
:: ------------------------------------------------------
echo.
echo ======================================================
echo [STEP 10] Getting Behavython repository
echo ======================================================

if exist "%TARGET_DIR%\.git" (
    echo [INFO] Repository already exists. Pulling latest changes...
    pushd "%TARGET_DIR%"
    git pull
    if errorlevel 1 (
        echo [ERROR] Git pull failed.
        popd
        pause
        exit /b 1
    )
    popd
) else (
    if exist "%TARGET_DIR%" (
        echo [ERROR] Target directory exists but is not a git repository:
        echo         %TARGET_DIR%
        echo Remove or rename it, then run again.
        pause
        exit /b 1
    )

    git clone "%REPO_URL%" "%TARGET_DIR%"
    if errorlevel 1 (
        echo [ERROR] Git clone failed.
        pause
        exit /b 1
    )
)

:: ------------------------------------------------------
:: STEP 11 - Final info
:: ------------------------------------------------------
echo.
echo ======================================================
echo Setup complete
echo ======================================================
echo.
echo Run Behavython with:
echo   cd /d "%TARGET_DIR%"
echo   "%CONDA_CMD%" run -n %ENV_NAME% python by_front.py
echo.
pause
exit /b 0
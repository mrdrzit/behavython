# Behavython — Installation and Execution Guide

> **Important:** There are two supported installation methods:
> - Installer (`.bat`) → requires using `run_behavython.bat` to launch.
> - `pip install` → use the `behavython` CLI command directly.

---

## 1. Prerequisites

Ensure the following components are installed and properly configured before proceeding:

* **Conda Distribution**
  One of the following must be installed and available in your system:
  * Miniforge3
  * Mambaforge
  * Miniconda3
  
  *Expected installation path:* `%USERPROFILE%\miniforge3` (or equivalent)

* **Git**
  Git must be installed and accessible via the system `PATH`. Verify with:
  ```cmd
  git --version
  ```

---

## 2. What the Installer Does

The installation script provisions a fully configured scientific environment:

* **Environment Setup:** Creates a Conda environment named `behavython` with Python 3.10.18.
* **GPU / ML Stack:** Installs CUDA Toolkit 11.2, cuDNN 8.1.0, and a PyTorch build compatible with CUDA 11.8.
* **Behavioral Analysis:** Installs DeepLabCut 2.3.10 (with GUI support).
* **Project Setup:** Clones or updates the repository to `%USERPROFILE%\Documents\behavython`.

---

## 3. Running the Installer (Windows CMD)

Follow these steps exactly to run the installer script:

1. Open Command Prompt (`Win + R` → `cmd` → `Enter`).
2. Navigate to the installer location (assuming it is in Downloads):
   ```cmd
   cd %USERPROFILE%\Downloads
   ```
3. Execute the installer by typing the name of the `.bat` file:
   ```cmd
   behavython_installer.bat
   ```

4. Wait for completion:

   * The script will:

     * Create the environment
     * Install dependencies
     * Validate imports
     * Check GPU availability
   * Do **not** close the terminal until you see:

     ```
     Setup complete
     ```

## 4. Launching Behavython

### A. If Installed via Installer (`.bat`)

The installer sets up a controlled environment and a dedicated launcher script. You **must use the provided launcher** to ensure the correct environment and dependencies are used.

1. Navigate to the launcher directory:
   ```cmd
   cd /d "%USERPROFILE%\Documents\behavython\src"
   ```
2. Run the launcher:
   ```cmd
   run_behavython.bat
   ```

*Note: This script will automatically detect your Conda/Mamba installation, validate the `behavython` environment, and launch the application using the correct Python environment. Direct execution via `python` or `conda run` is not recommended in this installation mode, as it may bypass environment checks.*

### B. If Installed via `pip`

If Behavython was installed using `pip install behavython`, you can launch it directly from any terminal, provided that you have activated the conda environment and installed the dependencies correctly as follows:

```cmd
mamba activate behavython
behavython
```

This uses the CLI entry point defined in the package and does **not** require navigating to any specific directory.
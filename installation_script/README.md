# Behavython — Installation and Execution Guide

## 3. Running the Installer (Windows CMD)

> **Important:** There are two supported installation methods:
> - Installer (`.bat`) → requires using this `run_behavython.bat`
> - `pip install` → use `behavython` command directly

## 1. Prerequisites

Ensure the following components are installed and properly configured before proceeding:

* **Conda Distribution**
  One of the following must be installed and available in your system:

  * Miniforge3
  * Mambaforge
  * Miniconda3
    Expected installation path:

  ```
  %USERPROFILE%\miniforge3 (or equivalent)
  ```

* **Git**
  Git must be installed and accessible via the system `PATH`.
  Verify with:

  ```cmd
  git --version
  ```

---

## 2. What the Installer Does

The installation script provisions a fully configured scientific environment:

* **Environment Setup**

  * Creates a Conda environment named `behavython`
  * Uses Python **3.10.18**

* **GPU / ML Stack**

  * CUDA Toolkit **11.2**
  * cuDNN **8.1.0**
  * PyTorch build compatible with CUDA **11.8**
  * TensorFlow-compatible stack

* **Behavioral Analysis**

  * Installs **DeepLabCut 2.3.10** (with GUI support)

* **Project Setup**

  * Clones or updates the repository:

    ```
    https://github.com/mrdrzit/behavython.git
    ```
  * Target directory:

    ```
    %USERPROFILE%\Documents\behavython
    ```

---

Follow these steps exactly if a double click on the `.bat` file does not work:

1. Open Command Prompt:

   ```
   Win + R → cmd → Enter
   ```

2. Navigate to the installer location from the terminal:

   ```cmd
   cd %USERPROFILE%\Downloads
   ```

3. Execute the installer from the terminal typing the name of the `.bat` file while in the correct directory:

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

---

Here is a revised **Section 4 (Launching Behavython)** with your requirement integrated clearly and unambiguously, plus a small structural improvement to distinguish install methods.

---

## 4. Launching Behavython

### A. If Installed via Installer (`.bat`)

The installer sets up a controlled environment and a dedicated launcher script.
You **must use the provided launcher** to ensure the correct environment and dependencies are used.

1. Navigate to the launcher directory:

```cmd
cd /d "%USERPROFILE%\Documents\behavython\src"
```

2. Run the launcher:

```cmd
run_behavython.bat
```

This script will:

* Automatically detect your Conda/Mamba installation
* Validate the `behavython` environment
* Launch the application using the correct Python environment
* Fallback to system Python if needed (with warnings)

> Direct execution via `python` or `conda run` is not recommended in this installation mode, as it may bypass environment checks and lead to runtime errors.

---

### B. If Installed via `pip`

If Behavython was installed using:

```cmd
pip install behavython
```

You can launch it directly from any terminal:

```cmd
behavython
```

This uses the CLI entry point defined in the package and does **not** require navigating to any directory.

---
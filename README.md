[![Issues](https://img.shields.io/github/issues/mrdrzit/Behavython)](https://github.com/mrdrzit/Behavython/issues)
[![Closed Issues](https://img.shields.io/github/issues-closed/mrdrzit/Behavython?color=21e00b)](https://github.com/mrdrzit/Behavython/issues?q=is%3Aissue+is%3Aclosed)
[![License](https://img.shields.io/github/license/mrdrzit/Behavython)](https://github.com/mrdrzit/Behavython/blob/main/LICENSE)

<br/>

<div align="center">
  <img src="https://raw.githubusercontent.com/mrdrzit/Behavython/main/logo/logo.png" width="300" alt="Behavython logo"/>

  <p>
    <strong>Automated Behavioral Analysis Interface</strong><br/>
    Currently built to integrate seamlessly with <a href="https://deeplabcut.github.io/DeepLabCut/README.html">DeepLabCut</a><br/>
    Other tools coming soon!
  </p>

  <p>
    <a href="https://github.com/mrdrzit/Behavython/issues">Report Bug</a>
    ·
    <a href="https://github.com/mrdrzit/Behavython/issues">Request Feature</a>
  </p>
</div>

---

## Table of Contents
- [About](#about)
- [Getting Started](#getting-started)
  - [Automated Installation (Windows)](#automated-installation-windows)
  - [Step-by-Step Installation (Manual)](#step-by-step-installation-manual)
  - [GPU Setup (Critical)](#gpu-setup-critical)
- [Pretrained Models](#pretrained-models)
- [Workflow & Configuration](#workflow--configuration)
- [Outputs & Metrics](#outputs--metrics)
- [Contributing](#contributing)
- [License](#license)
- [Contact](#contact)

---

## About

Behavython is an automated, high-throughput behavioral analysis interface. It features a modular architecture that converts raw tracking data (CSV/H5) into structured, biologically relevant behavioral metrics.

It natively supports:
- **Geometry-based paradigms:** Open Field, Elevated Plus Maze (EPM)
- **Interaction-based paradigms:** Social Recognition, Social Discrimination, Object Discrimination

The software enforces strict scientific reproducibility through automated environment management, robust data validation, and granular metric exports.

---

## Getting Started

### Automated Installation (Windows)

For Windows users, we provide a batch script that automates the creation of the Conda environment, installs the correct pip dependencies, and handles the CUDA setup. 

Download and run the installer from the repository:
- Open the link and save the page (right-click → Save As) as a `.bat` file, that is, `behavython_installer.bat`: <br>
[Download the file from here](https://github.com/mrdrzit/behavython/raw/main/installation_script/behavython_installer.bat)

- Learn how to use the installer from here:<br>
[Installation Guide](https://github.com/mrdrzit/behavython/tree/main/installation_script)

### Step-by-Step Installation (Manual)

We strongly recommend using Conda (or Miniconda/Mamba) to manage your isolated Python environment to prevent dependency conflicts.

**1. Create the environment:**
Create a clean environment explicitly using Python 3.10.
```bash
conda create -n behavython python=3.10
````

**2. Activate the environment:**
You must be inside the environment for the next steps.

```bash
conda activate behavython
```

**3. Install Behavython:**
Use `pip` inside the activated environment. **Crucial:** You must include the extra index URL to fetch the correct GPU-compiled PyTorch wheels. Omitting this may result in an incompatible CPU-only installation.

```bash
pip install behavython --extra-index-url [https://download.pytorch.org/whl/cu118](https://download.pytorch.org/whl/cu118)
```

### GPU Setup (Critical)

> ⚠️ **Required for correct execution**

Behavython relies heavily on GPU-accelerated frameworks. You must install the NVIDIA CUDA toolkit and cuDNN **AFTER** you have completed the pip installation above, and it must be done **INSIDE** the active `behavython` environment.

```bash
# Ensure you are still inside the 'behavython' environment
conda install -c conda-forge cudatoolkit=11.2 cudnn=8.1.0
```

Failure to configure the GPU dependencies correctly will result in severe processing slowdowns and potential runtime memory errors during DeepLabCut operations.

-----

## Pretrained Models

Behavython comes with a pretrained model for roi and mouse tracking (c57 black on white arena top view) These can be downloaded from the [GitHub Releases page](https://www.google.com/search?q=https://github.com/mrdrzit/Behavython/releases).

The latest release (`models-v1.0`) includes:

  * `c57_network_2025_minified.zip` - Optimized for C57 rodent tracking.
  * `roi_network.zip` - Dedicated network for Region of Interest detection in social recognition experiments.

Extract these ZIP archives into .behavython/models/ folder in your home directory located in: <br>
- **Windows:** `C:\Users\<YourUsername>\.behavython\models\`
- **Linux/Mac:** `~/.behavython/models/`
- **Note:** Ensure the extracted model files are directly within the `models` folder, not nested inside additional subdirectories.

-----

## Workflow & Configuration

### The Pipeline

```
Video → DeepLabCut → Filtered CSV → Behavython Core Pipeline → Metrics & Parquet
```

### Arena Configuration (Geometry Tasks)

For Open Field and Elevated Plus Maze (EPM) experiments, Behavython requires a `.json` configuration file defining the spatial geometry. You can extract these coordinates using the ImageJ Point Tool.

  * **Open Field:** Requires exactly 4 ordered corners (Top-Left, Top-Right, Bottom-Right, Bottom-Left).
  * **Elevated Plus Maze:** Requires exactly 12 ordered points defining the outer boundaries, arms, and center zone.

*Note: Incorrect coordinate ordering will corrupt spatial occupancy metrics.*

### ROI Configuration (Interaction Tasks)

For interaction paradigms, utilize the ImageJ Oval Tool to define your regions of interest.

  * Analyze → Set Measurements: Ensure **Centroid** and **Bounding Rectangle** are checked.
  * Name the exported CSV files logically (e.g., `video_roiL.csv` and `video_roiR.csv` for two-choice tasks, or `video_roi.csv` for single-object).

-----

## Outputs & Metrics

The pipeline outputs data in multiple formats for both statistical aggregation and granular programmatic review.

### General Storage

  * **`analysis_summary.xlsx` / `.csv`**: Scalar metrics aggregated per animal.
  * **`analysis_timeseries.parquet`**: Highly compressed, frame-by-frame data utilizing Apache Arrow for efficient downstream data science workflows.
  * **`analysis_log.json`**: Comprehensive error and warning logs per session.

### Task-Specific Metrics

**Geometry Experiments (Open Field, EPM):**

  * Zone occupancy times and percentages.
  * Transition counts between zones.
  * Discrete `spatial_state` arrays per frame.

**Interaction Experiments (Social/Object):**

  * Investigation and approach proportions.
  * Mean inter-bout intervals (seconds).
  * Discrete bout analysis (Collision bouts, approach-only bouts, abortive retreats).
  * `analysis_collisions.parquet`: Granular export containing distance to objects, interaction angles, and frame-by-frame flags.

-----

## Contributing

Contributions to the codebase and scientific pipelines are welcome.

  * Open an issue to report bugs or suggest features.
  * Submit a pull request for code changes. Ensure modifications align with the project's modular architecture and strict type-hinting standards.

-----

## License

This project is distributed under the GNU GPL v3.0 License. See the `LICENSE` file for more information.

-----

## Contact

  * **Matheus Costa** — [matheuscosta3004@gmail.com](mailto:matheuscosta3004@gmail.com)
  * **João Pedro** — [mcjpedro@gmail.com](mailto:mcjpedro@gmail.com)

-----

## Acknowledgments

  * Flávio Mourão — [https://github.com/fgmourao](https://github.com/fgmourao)
  * Núcleo de Neurociências (NNC) — [http://www.nnc.ufmg.br](http://www.nnc.ufmg.br)

<br>
<p align="left">
  <strong>Developed at</strong><br>
  Núcleo de Neurociências (NNC)<br>
  Universidade Federal de Minas Gerais (UFMG)<br>
  Brazil
</p>

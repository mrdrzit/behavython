[![Issues](https://img.shields.io/github/issues/mrdrzit/Behavython)](https://github.com/mrdrzit/Behavython/issues)
[![Closed Issues](https://img.shields.io/github/issues-closed/mrdrzit/Behavython?color=21e00b)](https://github.com/mrdrzit/Behavython/issues?q=is%3Aissue+is%3Aclosed)
[![License](https://img.shields.io/github/license/mrdrzit/Behavython)](https://github.com/mrdrzit/Behavython/blob/main/LICENSE)

<br/>

<div align="center">
  <img src="https://raw.githubusercontent.com/mrdrzit/behavython/refs/heads/main/src/behavython/gui/assets/images/logo.png" width="300" alt="Behavython logo"/>

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

For a streamlined setup, we provide a batch script that automates the entire process: creating the Conda environment, installing the required `pip` dependencies, and configuring the CUDA environment.

**How to use the installer:**

1.  **Download the script:** [**Download behavython_installer.bat**](https://github.com/mrdrzit/behavython/raw/main/installation_script/behavython_installer.bat)<br>
    *_Right-click the link and select **"Save Link As..."** to download it as a `.bat` file_*<br>
    *_In the windows explorer dowload prompt make sure to select "Save as type: All Files" and not "Text Documents" to avoid saving it as a `.txt` file_*
2.  **Run the Installer:** Double-click the downloaded file and follow the on-screen instructions.
3.  **Detailed Instructions:** For a full walkthrough of the automated process, refer to our [**Installation Guide**](https://github.com/mrdrzit/behavython/tree/main/installation_script).

### Step-by-Step Installation (Manual)

We strongly recommend using mamba (or Miniconda/Mamba) to manage your isolated Python environment to prevent dependency conflicts.

**1. Create the environment:**
Create a clean environment explicitly using Python 3.10.
```bash
mamba create -n behavython python=3.10
````

**2. Activate the environment:**
You must be inside the environment for the next steps.

```bash
mamba activate behavython
```

**3. Install Behavython:**
Use `pip` inside the activated environment. **Crucial:** You must include the extra index URL to fetch the correct GPU-compiled PyTorch wheels. Omitting this may result in an incompatible CPU-only installation.

```bash
pip install behavython --extra-index-url https://download.pytorch.org/whl/cu118
```

### GPU Setup (Critical)

> ⚠️ **Required for correct execution**

Behavython relies heavily on GPU-accelerated frameworks. You must install the NVIDIA CUDA toolkit and cuDNN **AFTER** you have completed the pip installation above, and it must be done **INSIDE** the active `behavython` environment.

```bash
# Ensure you are still inside the 'behavython' environment
mamba install -c conda-forge cudatoolkit=11.2 cudnn=8.1.0
```

Failure to configure the GPU dependencies correctly will result in severe processing slowdowns and potential runtime memory errors during DeepLabCut operations.

-----

## Pretrained Models

Behavython comes with a pretrained model for roi and mouse tracking (c57 black on white arena top view) These can be downloaded from the [GitHub Releases page](https://github.com/mrdrzit/behavython/releases/tag/models-v1.0).

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

**Example: Open Field (`arena.json`)**
Use the image as reference to order the points correctly.
Here, you can copy this example and save as a .json file to use 
In order, the points should be created in imageJ in this sequence:
1. top-left 
2. top-right
3. bottom-right
4. bottom-left

```json
{
  "experiment_type": "open_field",
  "arena_corners": [
    [126.0, 72.0],
    [637.0, 74.0],
    [633.0, 581.0],
    [128.0, 581.0]
  ]
}
```

**Example: Elevated Plus Maze (`arena.json`)**
As with the open field, use the image as reference to order the points correctly.
Here, you can copy this example and save as a .json file to use 
In order, the points should be created in imageJ in this sequence:
1. Center Top Left 
2. Top Left
3. Top Right
4. Center Top Right
5. Right Top
6. Right Bottom
7. Center Bottom Right
8. Bottom Right
9. Bottom Left
10. Center Bottom Left
11. Left Bottom
12. Left Top

```json
{
  "experiment_type": "elevated_plus_maze",
  "maze_points": [
    [128, 336],
    [384, 336],
    [384, 32],
    [514, 32],
    [514, 336],
    [900, 336],
    [900, 446],
    [514, 446],
    [514, 1042],
    [384, 1042],
    [384, 446], 
    [128, 446]
  ]
}
```

*Note: Incorrect coordinate ordering will corrupt spatial occupancy metrics.*

<div align="center">
  <img src="https://raw.githubusercontent.com/mrdrzit/behavython/refs/heads/main/src/behavython/gui/assets/images/geometry_reference.png" width="400" alt="Geometry Point Ordering Reference"/>
  <br>
  <em>Coordinate ordering reference for Open Field and Elevated Plus Maze.</em>
</div>

### ROI Configuration (Interaction Tasks)

For interaction paradigms (like Social Recognition or Object Discrimination), Behavython requires Region of Interest (ROI) definitions exported as CSV files. You can generate these using the ImageJ Oval Tool.

**ROI Requirements:**

  * **ImageJ Measurements:** Go to *Analyze → Set Measurements*. You **must** ensure both **Centroid** and **Bounding Rectangle** are checked. The resulting CSV needs the bounding box (BX, BY, Width, Height) and centroid (X, Y) coordinates.
  * **File Naming Convention:** The exported CSV must contain the original video name, followed by an ROI identifier:
    * *Single-Object/Target:* `video_name_roi.csv`
    * *Two-Choice Tasks:* `video_name_roiL.csv` and `video_name_roiR.csv` (Left and Right).
  * **File Format:** The file must be a standard `.csv` (Comma Separated Values) file.

**Example: Social Recognition/Discrimination (`video_name_roi.csv`)**
```csv
 ,Area,X,Y,BX,BY,Width,Height
1,24065,376.500,387.500,289,300,175,175
```

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

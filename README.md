[![Issues](https://img.shields.io/github/issues/mrdrzit/Behavython)](https://github.com/mrdrzit/Behavython/issues)
[![Closed Issues](https://img.shields.io/github/issues-closed/mrdrzit/Behavython?color=21e00b)](https://github.com/mrdrzit/Behavython/issues?q=is%3Aissue+is%3Aclosed)
[![License](https://img.shields.io/github/license/mrdrzit/Behavython)](https://github.com/mrdrzit/Behavython/blob/main/LICENSE)

<br/>

<div align="center">
  <img src="https://raw.githubusercontent.com/mrdrzit/Behavython/main/logo/logo.png" width="300" alt="Behavython logo"/>

  <p>
    <strong>Behavioral Analysis Interface</strong><br/>
    Currently built to integrate with <a href="https://deeplabcut.github.io/DeepLabCut/README.html">DeepLabCut</a><br/>
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
  - [Prerequisites](#prerequisites)
  - [Installation](#installation)
  - [GPU Setup (Critical)](#gpu-setup-critical)
- [Workflow](#workflow)
- [Outputs](#outputs)
- [Contributing](#contributing)
- [License](#license)
- [Contact](#contact)

---

## About

Behavython is an automated behavioral analysis interface designed to work alongside DeepLabCut (DLC). It converts raw tracking data (CSV/H5) into structured behavioral metrics.

It supports:
- Geometry-based paradigms (Open Field, Elevated Plus Maze)
- Interaction-based paradigms (Social/Object recognition)

The goal is reproducible, high-throughput behavioral quantification.

---

## Getting Started

### Prerequisites

- Python **3.10.x**
- NVIDIA GPU (**strongly recommended**)

---

### Installation

#### Pip

```bash
pip install behavython --extra-index-url https://download.pytorch.org/whl/cu118
````

If you omit the PyTorch index, you may install an incompatible CPU-only build.

---

#### Windows (Automated)

Use the provided:

```bat
run_behavython.bat
```

---

### GPU Setup (Critical)

> ⚠️ Required for correct execution

Behavython depends on GPU-accelerated frameworks. You must install CUDA + cuDNN after setting up your Python environment:

```bash
conda (or mamba) install -c conda-forge cudatoolkit=11.2 cudnn=8.1.0
```

Failure here leads to:

* Extremely slow processing
* Potential runtime errors

> ⚠️ This must be done **after** creating your Python environment, not globally. That is, inside of your `behavython` environment.

---

## Workflow

Pipeline:

```
Video → DeepLabCut → CSV → Behavython → Metrics
```

---

### Experiment Types

**ROI-Based**

* `social_recognition`
* `social_discrimination`
* `object_discrimination`

**Geometry-Based**

* `open_field`
* `elevated_plus_maze`

---

### Arena Configuration (Geometry)

Requires `.json` with coordinates extracted from ImageJ.

#### Open Field

4 points (ordered):

1. Top Left
2. Top Right
3. Bottom Right
4. Bottom Left

#### Elevated Plus Maze

12 ordered points:

* Outer: TL, TR, BL, BR
* Arms: LT, LB, RT, RB
* Center: CTL, CTR, CBL, CBR

> ⚠️ Incorrect ordering corrupts all spatial metrics

---

### ROI Experiments

Use ImageJ:

1. Oval Tool (hold Shift)
2. Analyze → Set Measurements:

   * Centroid
   * Bounding Rectangle
3. Press `Ctrl + M`
4. Save CSV

Naming:

* Single: `video_roi.csv`
* Dual: `video_roiL.csv`, `video_roiR.csv`

---

### DeepLabCut Processing

Use the **DEEPLABCUT** tab:

* Select `config.yaml`
* Select videos
* Click ANALYZE

> Without GPU: expect severe slowdown

---

### Running Analysis

In **ANALYSIS** tab:

* Set arena size (cm)
* Set FPS

Options:

* **Trim**: skips initial seconds
* **Crop**: limits total duration

Steps:

1. Select input files (CSV, ROI, videos)
2. Select output folder
3. (If needed) select `.json`

---

## Outputs

### General

* `analysis_summary.xlsx / .csv`
  Scalar metrics per animal

* `analysis_timeseries.parquet`
  Frame-by-frame data

---

### Geometry Experiments

Adds:

* Zone occupancy
* Transition counts
* `spatial_state` per frame

---

### ROI Experiments

Adds:

* Investigation ratio
* Approach rate
* Retreat rate

File:

* `analysis_collisions.parquet`

  * Distance to object
  * Angles
  * Interaction flags

---

## Contributing

Contributions are welcome.

* Open an issue
* Submit a pull request
* Suggest improvements

---

## License

GNU GPL v3.0
See `LICENSE` file.

---

## Contact

* Matheus Costa — [matheuscosta3004@gmail.com](mailto:matheuscosta3004@gmail.com)
* João Pedro Carvalho Moreira — [mcjpedro@gmail.com](mailto:mcjpedro@gmail.com)

---

## Acknowledgments

* Flávio Mourão — [https://github.com/fgmourao](https://github.com/fgmourao)
* Núcleo de Neurociências (NNC) — [http://www.nnc.ufmg.br](http://www.nnc.ufmg.br)

---

## Developed at

Núcleo de Neurociências (NNC)
Universidade Federal de Minas Gerais (UFMG)
Brazil

[issues-shield]: https://img.shields.io/github/issues/mrdrzit/Behavython 
[issues-url]: https://github.com/mrdrzit/Behavython/issues 
[closed-issues-shield]: https://img.shields.io/github/issues-closed/mrdrzit/Behavython?color=%2321e00b 
[closed-issues-url]: https://github.com/mrdrzit/Behavython/issues?q=is%3Aissue+is%3Aclosed 
[license-shield]: https://img.shields.io/github/license/mrdrzit/Behavython 
[license-url]: https://github.com/mrdrzit/Behavython/blob/main/LICENSE
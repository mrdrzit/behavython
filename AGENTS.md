# Repository Guidelines

Single source of truth for AI agents. `CLAUDE.md` and `GEMINI.md` only import this file.
Detailed conventions live in `docs/dev/` and are loaded on demand through the skills in `.claude/skills/` (mirrored in `.agents/skills/`; keep both copies identical).

## 1. Project Shape
- Package: `src/behavython`, Python 3.10 only. Entry points: `behavython` (PySide6 GUI, `main.py`) and `behavython-cli` (`cli.py`).
- Layers: `core/` (defaults, paths, exceptions) -> `pipeline/` (metrics, geometry, filters, plotting, export, `plugins/dlc.py`) -> `services/` -> `tasks/` (QThread workers) -> `gui/`.
- `pipeline/` and `services/` must not import from `gui/`. Long-running GUI work goes through `tasks/TaskRunner`, never the main thread.
- Every user-facing feature exists in both GUI and CLI. When changing one, check the other.
- Constants (colors, thresholds, defaults) live in `core/defaults.py`. Do not hardcode new ones in modules.
- `src/behavython/_legacy/` is reference only. Do not import from it or edit it.
- `.ui` files are Qt Designer XML. Make minimal edits and keep widget `objectName`s stable; `main_window.py` references them.

## 2. Data Safety
- Never modify user data: DLC tracking files (`.h5`, `.csv`), videos, ROI files, or DLC `config.yaml`, unless the task is explicitly about that.
- Broken DLC configs are repaired into a copy (`prepare_dlc_config`); keep that pattern.
- Scripts that mutate DLC projects (`scripts/`, `src/behavython/scripts/`) must support a dry run or write to a new path.

## 3. Environment
- The only Python interpreter allowed is the miniforge `behavython` env: `%USERPROFILE%\miniforge3\envs\behavython\python.exe` (PowerShell: `$HOME\miniforge3\envs\behavython\python.exe`; Git Bash: `$HOME/miniforge3/envs/behavython/python.exe`).
- NEVER call `python`, `python3`, `py`, or `pip` bare, in any shell or tool command. On this machine they resolve to the system install or the Microsoft Store stub, not the project env.
- Always call the interpreter by full path, and run pip as a module of it: `<env python> -m pip ...`.
- The package is not installed into the env. Run project code from the repo root with `PYTHONPATH=src` (PowerShell: `$env:PYTHONPATH = "src"`).
- Run env tools by full path too: `$HOME\miniforge3\envs\behavython\Scripts\ruff.exe` and `pyright.exe`, or as `<env python> -m ruff`.
- If that interpreter is missing or fails to start, stop and ask. Do not fall back to another Python.
- DeepLabCut is optional (`tf` and `torch` extras). Import it only through `plugins/dlc.py:load_deeplabcut()`; the core pipeline must run without it.
- Ask before installing or upgrading packages. Respect `numpy<2.0`.

## 4. Communication
- Short, direct, answer first.
- Plain typography: no em-dashes, en-dashes, double dashes, or smart quotes.
- Do not hard-wrap sentences in markdown files.
- Ban AI marketing adjectives, filler verbs, and empty intensifiers.

## 5. Tools
- `rg` for text search, `ast-grep` for structural search and refactors, native file tools for reading and editing.
- Use targeted reads for large files (`main_window.py`, `dlc.py`, `plotting.py`, `dlc_assisted_label_session.py`, `cli.py`, `metrics.py`).
- Use the least expensive tool that provides sufficient information.
- Do not write ad hoc Python or PowerShell scripts solely to search or inspect the repository.
- Prefer targeted reads (`Read` with `offset` and `limit`) around the target function, class, or section.
- For files exceeding 200 lines, inspect targeted ranges first.
- Read complete files only when necessary to understand module state, pipeline flow, or class invariants. Never guess unseen code.

## 6. Formatting and Validation
- Ruff is the formatter and linter (`ruff.toml`, line length 121). Run `ruff format` and `ruff check` on every Python file you modify.
- Introduce no new `ruff check` errors. Do not mass-fix pre-existing lint errors unless asked.
- Report what was verified and what was not.

## 7. Modification Protocol
Before: locate the implementation and its callers in both GUI and CLI paths, then pick the smallest sufficient diff.
After: review `git diff`, run ruff on changed files, smoke-test, and check downstream consumers if a result key, export column, or signature changed (`export.py`, `plotting.py`, README outputs section).

## 8. Domain Guides
- Metrics, units, filters, and analysis windows: [docs/dev/metrics_conventions.md](docs/dev/metrics_conventions.md). Changing a metric definition changes numbers users may have published.
- Plots and videos: [docs/dev/plotting_conventions.md](docs/dev/plotting_conventions.md). Describes current practice; no restyling unless asked.

## 9. Network Creation
- Location and name: `E:\Matheus\Deeplabcut_networks\<Task>-<SCORER>-<YYYY-MM-DD>`, for example `CD1-GUSTAVO-2026-10-12`. Check existing folders there before naming.
- Do not use `deeplabcut.create_new_project` for videos that must stay in place: DLC 3.0.2 moves them when symlinks fail. Build the folder with a script (`scripts/make_dlc_network.py` is the reference). Never modify the source video folders.
- `video_sets` in `config.yaml` decides which `labeled-data` folders enter training. Folders without an entry are skipped silently. Each entry points at the original video path with its real crop size.
- Holdout and test videos stay out of `video_sets`. Their frames may sit in `labeled-data`, but they are labeled for evaluation only.
- Frame picking: skip the first 10% of each video (the animal is being placed). Hard frames come from the unsure or disagreeing frames of two networks, not from the network under test. Use at least 30 frames between picks.
- Pre-label with a network other than the one being evaluated. If the corrections move almost no points, the labels are not independent and cannot rank that network.
- Review with `scripts/dlc_review.py` (one napari process per folder, resumable through `review_log.csv`). napari-deeplabcut only saves folders inside `labeled-data`, and reusing one viewer for many folders runs out of memory.
- What counts as what: adding frames to the same network is a new `iteration` (set `iteration: N+1`, build the dataset with the previous test images so results stay comparable). A different backbone is a new network: a new project folder with its own date, the same labeled frames and the same test images, built with `scripts/make_dlc_network.py`. Always use shuffle 1, because behavython does not pass a shuffle to DeepLabCut. A different bodypart set or a retrain from unrelated data is also a new project folder. State which of these applies before creating anything.
- The recipe that matches DLC 2.3 TensorFlow (`CD1-GUSTAVO-2026-10-15`) is `scripts/make_dlc_network.py <source> --date YYYY-MM-DD --recipe tf`: resnet_50 with the SuperAnimal-TopViewMouse backbone, whole frames resized to 0.8 of the median frame side (576 for CD1), plateau target with BCE loss and sigmoid output, light affine, no flips, 300 epochs. The DLC 3 default (Gaussian target, MSE, no sigmoid) caps likelihood near 0.9 and leaves many frames unsure. With the BCE head, likelihood rarely flags wrong points, so check jumps and review frames. `make_dlc_network.py` sets `runner.key_metric: test.rmse` with `key_metric_asc: false`, so `snapshot-best` is the lowest test RMSE; DLC's default `test.mAP` saturates at 100 on small test sets and then never updates. On a few test images, RMSE differences under about 0.5 px are noise: confirm on holdout frames. Projects built before this change (up to and including `find_cup_circle-matheus-2026-10-19`) still rank by mAP.
- Fine-tuning from a snapshot: resume from a copy with the optimizer and scheduler state removed (`metadata` and `model` only), start at a lower rate (1e-4 for resnet_50), and write absolute epoch milestones, because the epoch counter continues from the snapshot. `PoseConfig` rejects unknown `runner` keys such as `load_scheduler_state_dict`.
- Moving or copying a project to another drive or machine: run `python fix_project_paths.py` inside the project folder (a copy is placed there by `make_dlc_network.py`; the source is `scripts/fix_project_paths.py`). It rewrites `project_path` in `config.yaml`, `metadata.project_path` and `metadata.pose_config_path` in `pytorch_config.yaml`, and `dataset` in `test/pose_cfg.yaml`. The `video_sets` keys stay as they are.
- HRNet needs every input to be a multiple of 32 on both sides, or it fails with "size of tensor a (138) must match the size of tensor b (140)". Keep `data.train.auto_padding` and `data.inference.auto_padding` (`pad_width_divisor` and `pad_height_divisor` of 32) from the hrnet template when copying another network's `data` section, and use a 544 training crop. Training crops hit the error at epoch 1, full frames (738 x 726 and similar) at the first evaluation. `make_dlc_network.py` does this for hrnet.
- Train with `scripts\train_dlc_network.bat <project>` (drag a project folder onto it, or pass a path). It runs `scripts/train_dlc_network.py` in the miniforge `behavython` env, trains exactly what `train/pytorch_config.yaml` says, and prints one line per epoch (loss, mAP and best mAP, time left). `--watch` attaches to a running training; it refuses a train folder with snapshots unless `--overwrite`. Needs DeepLabCut 3.0.2 (PyTorch) and PyYAML only, not behavython.
- Training can also run from the DLC GUI or a terminal `deeplabcut.train_network` call. The GUI Train tab overrides epochs, save epochs, snapshots to keep, and display iterations. It does not override the learning-rate schedule or augmentation, which live in `train/pytorch_config.yaml`.
- DLC 3.0.2 augmentation keys: affine, crop_sampling, hflip (with symmetries), hist_eq, motion_blur, covering, elastic_transform, grayscale, gaussian_noise. There is no brightness or contrast option.
- `scripts/profile_training.py <train or project folder> [--out DIR]` summarizes a finished training (config, time, test metrics per evaluation, snapshots present or rotated away). `scripts/profile_analysis.py <folders or .h5 files> [--out DIR]` summarizes tracking quality of analyzed videos per folder and scorer, without labels.
- Choose the snapshot on labeled holdout frames (`scripts/eval_holdout.py`), reported per phase (pre and pos pintura) and per bodypart, not on test mAP alone. Checkpoints within 0.3 mAP of each other are tied.
- Take a backup of `config.yaml` and the training dataset folder before any step that rewrites them, and verify that the test set is unchanged afterwards.
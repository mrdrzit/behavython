---
name: dlc-integration
description: Rules for DeepLabCut integration in behavython (video analysis, frame extraction, labeling sessions, config repair, TF and PyTorch backends). Use when editing pipeline/plugins/dlc.py, services/dlc_assisted_label_session.py, or DLC helper scripts.
---

# DeepLabCut Integration

## Key Rules
- DeepLabCut is optional. Import it only through `load_deeplabcut()` in `pipeline/plugins/dlc.py`, inside the function that needs it.
- Support both backends: DLC 2.3.x (TensorFlow, `tf` extra) and DLC 3.0+ (PyTorch, `torch` extra). Check backend-specific paths before assuming one.
- Never edit a user's `config.yaml` in place. Use `prepare_dlc_config()`, which returns a repaired copy when needed.
- Shuffle and training set index are inferred by `infer_dlc_shuffle_and_trainingsetindex()`; explicit user values (CLI `--shuffle`, `--trainingsetindex`) override it.
- DLC tasks take a request dataclass from `pipeline/models.py` and report through `progress`, `log`, and `warning` callbacks so they run in both GUI workers and the CLI.
- Scripts that modify DLC projects (labels, `CollectedData_*.h5`, training sets) need a dry run or must write to a new path.

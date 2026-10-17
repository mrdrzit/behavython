---
name: metrics-pipeline
description: Conventions for behavython metric computation, analysis windows, scaling, thresholds, and exports. Use when editing pipeline/metrics.py, workflow.py, filters.py, geometry.py, or export.py.
---

# Metrics Pipeline

Read [metrics_conventions.md](../../../docs/dev/metrics_conventions.md) before changing metric code.

## Key Rules
- Changing a threshold, window, or definition changes published results: flag it to the user and note it in release notes.
- Keep unit suffixes in result keys (`_s`, `_cm`, `_cm_s`, `_array`).
- A new or renamed metric must be updated in both exporters (`export_summary_metrics`, `export_results_to_parquet`) and the README outputs section.
- Analysis-window logic is duplicated in `preprocess_animal` and `analyze_maze_animal`; change both copies.
- Express time windows in seconds and convert with fps; do not add new frame-count constants.

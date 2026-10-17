---
name: plotting
description: Current conventions for behavython per-animal plots, likelihood plots, and annotated videos. Use when editing pipeline/plotting.py or adding a figure or animation.
---

# Plotting

Read [plotting_conventions.md](../../../docs/dev/plotting_conventions.md) before adding or changing plots.

## Key Rules
- Follow the current style; do not restyle existing plots unless asked.
- Frame-matched size at `dpi = 100`, PNG, `bbox_inches="tight"`, close every figure.
- Colors from `core/defaults.py` only; matplotlib/OpenCV pairs via `ZONE_STYLES`.
- File names: `"{animal_name} - <Description>.png"`; keep existing names stable.

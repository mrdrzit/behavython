# Plotting Conventions

Describes how `pipeline/plotting.py` works today. New plots follow these patterns; restyling existing plots is out of scope unless requested.

## 1. Setup
- Matplotlib runs on the `Agg` backend (set at the top of `plotting.py`). Never call `plt.show()` in pipeline code.
- `apply_scientific_style()` sets the global style: light-weight sans-serif fonts (Helvetica Neue Light first, DejaVu Sans last), axis labels and titles at 10 pt, legend at 8 pt, white legend frame without edge.
- Close every figure after saving (`plt.close(fig)`); batches run over many animals.

## 2. Size and Resolution
- Per-animal plots match the video frame: `figsize = (target_w_px / dpi, target_h_px / dpi)` with `dpi = 100`, so plots overlay the arena image one to one.
- Likelihood plots: `figsize = (12, 4)`, `dpi = 120`, y-axis fixed to 0-1, p-cutoff drawn as a dashed line.
- Save with `bbox_inches="tight"`. Output is PNG only.

## 3. File Naming
- Saved in the animal's output folder as `"{animal_name} - <Description>.png"`, e.g. `"{animal_name} - Distance accumulated over time.png"`.
- Keep existing names stable; users and downstream scripts look for them.

## 4. Colors
All colors come from `core/defaults.py`:
- Heatmaps and KDE: `PLOT_COLORMAP` (`inferno`).
- Trajectories: `PLOT_TRAJECTORY_COLOR` (orangered), `PLOT_TRAJECTORY_LINEWIDTH` (1.5); scatter `PLOT_SCATTER_SIZE` (6).
- Zones: `ZONE_STYLES[zone]` gives a paired `{"mpl": hex, "cv2": BGR tuple}`; unknown zones use `FALLBACK_ZONE_STYLE`.
- Likelihood plots: colormap from the DLC config (`colormap`, fallback `Set2`).
- OpenCV colors are BGR tuples (`CV2_*`). When adding a color used in both matplotlib and OpenCV, add it as a pair like `ZONE_STYLES`.
- `ZONE_COLOR_MAP`, `ZONE_FALLBACK_COLORS`, `PLOT_GEOMETRY_COLORS`, and `CV2_POLY_COLORS` are currently unused outside `_legacy/`; prefer `ZONE_STYLES` in new code.

## 5. Videos and Animations
- Two renderers exist: OpenCV (`_opencv_animate_*`, preferred for speed) and matplotlib `FuncAnimation` (`_matplotlib_animate_*`).
- FFmpeg is resolved with `core.utils.get_ffmpeg_path()`; do not assume it is on PATH.

## 6. Adding a Plot
1. Read data from the result dict; do not recompute metrics in plotting code.
2. Use the frame-matched size and the constants above.
3. Name the file with the `"{animal_name} - ..."` pattern and close the figure.
4. Respect `AnalysisOptions.plot_options` and `generate_video`; plots and videos are opt-in per request.

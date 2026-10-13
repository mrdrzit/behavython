"""
Reviews the pre-labeled folders of one round, one napari window per folder.

Folders are read from frames/frames_manifest.csv and opened from labeled-data, so run
`build_cd1_retrain_project.py collect` after pre-labeling and before reviewing.

For each folder a fresh napari process opens it with the DeepLabCut keypoint controls.
Correct the points, press Ctrl+S, close the window: the next folder opens.
A folder counts as reviewed when its CollectedData file was saved while the window was open
(if you close without saving, the terminal asks what to do).

Progress is kept in review_log.csv in the project, so stopping and re-running resumes where you stopped.
"""

import argparse
import csv
import subprocess
import sys
import warnings
from datetime import datetime
from pathlib import Path

LABEL_FILE = "CollectedData_GUSTAVO.h5"


def resolve_folders(project: Path, round_name: str) -> list[Path]:
    with open(project / "frames" / "frames_manifest.csv", encoding="utf-8") as handle:
        stems = sorted({row["stem"] for row in csv.DictReader(handle) if row["round"] == round_name})
    # napari-deeplabcut writes the parent folder name into the row index, so saving only works under labeled-data
    folders = []
    for stem in stems:
        candidate = project / "labeled-data" / stem
        if (candidate / LABEL_FILE).exists():
            folders.append(candidate)
        else:
            print(f"skip (not collected into labeled-data yet): {stem}")
    return folders


def read_log(log_path: Path) -> set[str]:
    if not log_path.exists():
        return set()
    with open(log_path, encoding="utf-8") as handle:
        return {row["folder"] for row in csv.DictReader(handle)}


def append_log(log_path: Path, round_name: str, folder: Path):
    is_new = not log_path.exists()
    with open(log_path, "a", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        if is_new:
            writer.writerow(["round", "folder", "reviewed_at"])
        writer.writerow([round_name, folder.name, datetime.now().isoformat(timespec="seconds")])


def open_single(folder: Path, title: str):
    """Child process: one viewer, one folder, exits when the window is closed."""
    warnings.filterwarnings("ignore", category=FutureWarning)
    import napari

    viewer = napari.Viewer(title=title)
    viewer.window.add_plugin_dock_widget("napari-deeplabcut", "Keypoint controls")
    viewer.open(str(folder), plugin="napari-deeplabcut")

    def start_in_select_mode():
        from napari.layers import Points

        points = [layer for layer in viewer.layers if isinstance(layer, Points)]
        if points:
            viewer.layers.selection.active = points[0]
            points[0].mode = "select"

    # Run once now and once after the plugin's own deferred setup, which may switch the layer to "add"
    from qtpy.QtCore import QTimer

    start_in_select_mode()
    QTimer.singleShot(500, start_in_select_mode)
    napari.run()


def review(project: Path, round_name: str, include_reviewed: bool):
    folders = resolve_folders(project, round_name)
    log_path = project / "review_log.csv"
    reviewed = read_log(log_path)
    queue = [f for f in folders if include_reviewed or f.name not in reviewed]
    print(f"{round_name}: {len(folders)} folders, {len(folders) - len(queue)} already reviewed, {len(queue)} to go.")

    for position, folder in enumerate(queue, start=1):
        done = len(read_log(log_path) & {f.name for f in folders})
        title = f"Review {round_name} [{position}/{len(queue)}] (reviewed {done}/{len(folders)}) - {folder.name}"
        while True:
            print(f"\n[{position}/{len(queue)}] {folder.name}: correct, Ctrl+S, close the window.")
            before = (folder / LABEL_FILE).stat().st_mtime
            subprocess.run([sys.executable, __file__, "--single", str(folder), title], check=False)
            if (folder / LABEL_FILE).stat().st_mtime > before:
                append_log(log_path, round_name, folder)
                break
            answer = input("Not saved. [r]eopen, mark as [o]k without changes, [s]kip, [q]uit: ").strip().lower()
            if answer == "o":
                append_log(log_path, round_name, folder)
                break
            if answer == "s":
                break
            if answer == "q":
                return
    print(f"\n{round_name}: {len(read_log(log_path) & {f.name for f in folders})}/{len(folders)} folders reviewed.")


def main():
    if len(sys.argv) >= 2 and sys.argv[1] == "--single":
        open_single(Path(sys.argv[2]), sys.argv[3])
        return

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("project", help="Path to the project folder (the one with config.yaml).")
    parser.add_argument("round", choices=["round1", "round2", "holdout"])
    parser.add_argument("--all", action="store_true", help="Also open folders already marked as reviewed.")
    args = parser.parse_args()
    review(Path(args.project), args.round, args.all)


if __name__ == "__main__":
    main()

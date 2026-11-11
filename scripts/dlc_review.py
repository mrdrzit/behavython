"""
Reviews DeepLabCut labeled-data folders of any project, one napari window per folder.

Each folder opens in a fresh napari process with the DeepLabCut keypoint controls, the keypoints
layer selected and the "select points" tool active. Correct, press Ctrl+S, close the window: the
next folder opens. A folder counts as reviewed when its CollectedData file was saved while its window
was open; if you close without saving, the terminal asks what to do.

Folders to review (combine freely; default = every folder in the target's labeled-data):
  names          folders already inside <target>/labeled-data, e.g. TR_A1_2026-06-21_09-30-00
  paths          folders anywhere else (another project, a pre-labeling output, ...). They are imported
                 first: copied into <target>/labeled-data, with labels converted to the target's
                 scorer and bodyparts. If the folder already exists in the target, only images it
                 lacks are added (with their labels); its existing images and labels stay as they are,
                 and its CollectedData file is backed up first.
  --list FILE    text file with one name or path per line (quotes around a path are ignored)
  --match GLOB   only folders whose name matches, e.g. "TT_*2026-07*"
A folder given as a path is imported only when it holds a CollectedData*.csv, a CollectedData*.h5 and
png/jpg/tif images; otherwise it is skipped with the reason.

video_sets: training reads only labeled-data folders that have an entry in config.yaml's video_sets, and
that entry stores the video's path. For each folder without one, the script looks for <folder name>.mp4
(or .avi, .mov, .mkv):
  default          next to the imported folder. It explains why, asks for a permanent folder, copies the
                   videos there and points video_sets at the copies.
  --videos-dir DIR in DIR, with no prompt and no copy: video_sets points at the videos in DIR, so they must
                   stay there.
config.yaml is backed up before it changes. --dry-run only reports what would be added.

Progress is kept in <target>/review_log.csv, so re-running resumes where you stopped.
napari-deeplabcut can only save folders that sit inside a project's labeled-data, hence the import.

Copying points to many frames (e.g. an ROI that does not move within a video):
  1. Correct one frame. Click an empty spot (clears the selection), drag a box around its points, Ctrl+C.
  2. D (next frame). Box all of that frame's points, Delete, then Ctrl+V. Check, and repeat on each frame.
     No need to copy again: the clipboard stays after a paste.
  - Paste only adds keypoints the frame does not have yet, and the clipboard keeps only what was pasted.
    Delete all points of a frame before pasting, or later frames get fewer points.
  - The selection spans the whole layer, not one frame. Clear it before Ctrl+C, or points selected in
    other frames are copied too. Shift+A selects every point of every frame; A and D change frames.
"""

import argparse
import csv
import fnmatch
import shutil
import statistics
import subprocess
import sys
import time
import warnings
from datetime import datetime, timedelta
from pathlib import Path

# How each napari window starts. Edit these to taste.
START_TOOL = "add"  # "select" (click to select, drag to move points), "add", or "pan_zoom"
LABEL_MODE = "quick"  # "sequential", "quick" (adding a placed point moves it), or "loop" (same point frame after frame)
POINT_SIZE = 10  # point size in pixels; None keeps the plugin's (from config.yaml dotsize)
START_LABEL = "Focinho"  # bodypart selected for the next point; None = the first bodypart in config.yaml
# True: every frame starts at START_LABEL. False: the plugin's choice, the first bodypart not yet placed in that frame.
START_LABEL_EACH_FRAME = True

IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".tif", ".tiff"}
VIDEO_SUFFIXES = (".mp4", ".avi", ".mov", ".mkv")


def clean_entry(text: str) -> str:
    return text.strip().strip("\"'").strip()


def invalid_reason(folder: Path) -> str | None:
    """Why a folder given as a path cannot be imported, or None if it is a labeled folder."""
    if not folder.is_dir():
        return "not a folder"
    missing = [pattern for pattern in ("CollectedData*.csv", "CollectedData*.h5") if not any(folder.glob(pattern))]
    if missing:
        return f"no {' or '.join(missing)}"
    if not any(p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES for p in folder.iterdir()):
        return "no png/jpg/tif images"
    return None


def read_project(config_path: Path) -> dict:
    import yaml

    with open(config_path, encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle)
    if cfg.get("multianimalproject"):
        raise SystemExit("Multi-animal projects are not supported.")
    return {"root": config_path.parent, "scorer": cfg["scorer"], "bodyparts": list(cfg["bodyparts"])}


def import_folder(source: Path, project: dict, dry_run: bool = False) -> Path | None:
    """Copies a labeled folder into the target's labeled-data, converting its labels to the target format.

    With dry_run, only reports what would be copied or merged and changes nothing.
    """
    import pandas as pd

    destination = project["root"] / "labeled-data" / source.name
    if destination.resolve() == source.resolve():
        return destination
    images = [p for p in source.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES]
    if destination.exists():
        # Merge: only images the target lacks are added; its existing images and labels are never touched.
        images = [p for p in images if not (destination / p.name).exists()]
        if not images:
            print(f"  already in target, nothing new: {source.name}")
            return destination
    if not images:
        print(f"  skip (no images): {source}")
        return None
    if dry_run:
        action = f"would merge {len(images)} new images into" if destination.exists() else f"would import {len(images)} images as"
        print(f"  --dry-run, {action} labeled-data/{source.name}")
        return destination

    labels = None
    h5_files = sorted(source.glob("CollectedData_*.h5"))
    if h5_files:
        df = pd.read_hdf(h5_files[0])
        missing = [bp for bp in project["bodyparts"] if bp not in df.columns.get_level_values("bodyparts")]
        if missing:
            print(f"  {source.name}: bodyparts not in source, left empty: {missing}")
        columns = pd.MultiIndex.from_product(
            [[project["scorer"]], project["bodyparts"], ["x", "y"]], names=["scorer", "bodyparts", "coords"]
        )
        values = {}
        for _, bodypart, coord in columns:
            match = [c for c in df.columns if c[1] == bodypart and c[2] == coord]
            values[(project["scorer"], bodypart, coord)] = df[match[0]] if match else float("nan")
        labels = pd.DataFrame({column: values[column] for column in columns}, index=df.index)
        labels.columns = columns
        image_names = [Path(str(idx[-1] if isinstance(idx, tuple) else idx)).name for idx in df.index]
        labels.index = pd.MultiIndex.from_arrays([["labeled-data"] * len(labels), [source.name] * len(labels), image_names])
        labels = labels[labels.index.get_level_values(2).isin({p.name for p in images})]

    label_file = destination / f"CollectedData_{project['scorer']}.h5"
    merged = label_file.exists()
    if merged:
        shutil.copy2(label_file, label_file.with_name(label_file.name + f".bak-{datetime.now().astimezone():%Y%m%d-%H%M%S}"))
        existing = pd.read_hdf(label_file)
        labels = existing if labels is None else pd.concat([existing, labels]).sort_index()

    destination.mkdir(parents=True, exist_ok=True)
    for image in images:
        shutil.copy2(image, destination / image.name)
    if labels is not None:
        labels.to_hdf(label_file, key="df_with_missing", mode="w")
        labels.to_csv(label_file.with_suffix(".csv"))
    action = f"merged {len(images)} new images" if merged else "imported"
    print(f"  {action}: {source} -> labeled-data/{source.name}" + ("" if labels is not None else " (no labels, starts empty)"))
    return destination


def resolve_folders(project: dict, entries: list[str], pattern: str | None, dry_run: bool = False) -> tuple[list[Path], dict[Path, Path]]:
    """Folders to review, and for each imported one the folder it was imported from."""
    labeled_data = project["root"] / "labeled-data"
    if not entries:
        entries = sorted(p.name for p in labeled_data.iterdir() if p.is_dir())
    folders, sources = [], {}
    for entry in entries:
        path = Path(entry)
        if not path.is_absolute() or path.parent == labeled_data:
            folder = labeled_data / path.name
            if not folder.is_dir():
                print(f"  skip (not in labeled-data): {entry}")
                continue
        else:
            reason = invalid_reason(path)
            if reason:
                print(f"  skip ({reason}): {path}")
                continue
            folder = import_folder(path, project, dry_run)
            if folder is None:
                continue
            sources[folder] = path
        if pattern and not fnmatch.fnmatch(folder.name, pattern):
            continue
        if folder.is_dir() and not any(p.suffix.lower() in IMAGE_SUFFIXES for p in folder.iterdir()):  # not yet imported on --dry-run
            print(f"  skip (no images): {folder.name}")
            continue
        if folder not in folders:
            folders.append(folder)
    return folders, sources


def find_video(base: Path, name: str) -> Path | None:
    return next((base / f"{name}{suffix}" for suffix in VIDEO_SUFFIXES if (base / f"{name}{suffix}").is_file()), None)


def choose_folder(title: str) -> Path | None:
    try:
        import tkinter
        from tkinter import filedialog

        root = tkinter.Tk()
        root.withdraw()
        root.attributes("-topmost", True)
        chosen = filedialog.askdirectory(title=title)
        root.destroy()
    except Exception:  # no display available: ask in the terminal
        chosen = clean_entry(input(f"{title}: "))
    return Path(chosen) if chosen else None


def copy_videos(videos: dict[Path, Path], destination: Path) -> dict[Path, Path]:
    """Copies each video into destination; an existing file of the same size is reused, a different one is kept."""
    destination.mkdir(parents=True, exist_ok=True)
    copied = {}
    for folder, video in videos.items():
        target = destination / video.name
        if target.exists() and target.stat().st_size != video.stat().st_size:
            print(f"  not copied, a different {target.name} is already in {destination}: {folder.name}")
            continue
        if not target.exists():
            print(f"  copying {video.name} ({video.stat().st_size / 1e6:.0f} MB)")
            shutil.copy2(video, target)
        copied[folder] = target
    return copied


def add_video_sets(config: Path, folders: list[Path], sources: dict[Path, Path], videos_dir: Path | None, dry_run: bool):
    """Adds a video_sets entry for each folder that has none, so training does not skip it."""
    import cv2
    from ruamel.yaml import YAML

    yaml = YAML()
    yaml.width = 4096  # long video paths with spaces get folded otherwise, which PyYAML (used by DLC) cannot read
    with open(config, encoding="utf-8") as handle:
        cfg = yaml.load(handle)
    known = {Path(str(video).replace("\\", "/")).stem for video in cfg.get("video_sets") or {}}
    missing = [folder for folder in folders if folder.name not in known]
    if not missing:
        return

    print(f"\n{len(missing)} folder(s) have no video_sets entry in {config.name}. Training reads only folders listed")
    print("there, and each entry stores the path of the folder's video, so that path must not change.")
    if videos_dir:
        print(f"--videos-dir: video_sets will point at the videos in {videos_dir} as they are (no copy).")
        print("Keep them there: moving or deleting them breaks the paths stored in config.yaml.")
        videos = {folder: find_video(videos_dir, folder.name) for folder in missing}
    else:
        videos = {folder: find_video(sources[folder].parent, folder.name) for folder in missing if folder in sources}
    videos = {folder: video for folder, video in videos.items() if video}
    found = set(videos)

    if videos and not videos_dir:
        for video in videos.values():
            print(f"  found {video}")
        if dry_run:
            print("--dry-run: would ask for a permanent folder, copy these videos there and add them to video_sets.")
            videos = {}
        else:
            input("These will be copied to a permanent folder you choose. Press Enter to choose it (cancel = skip): ")
            destination = choose_folder("Permanent folder for the videos referenced by video_sets")
            videos = copy_videos(videos, destination) if destination else {}
            if not destination:
                print("  no folder chosen: no video was copied and video_sets was not changed.")

    entries = {}
    for folder, video in videos.items():
        capture = cv2.VideoCapture(str(video))
        width, height = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)), int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        capture.release()
        if not width or not height:
            print(f"  cannot read the frame size of {video}: no entry for {folder.name}")
            continue
        entries[str(video)] = {"crop": f"0, {width}, 0, {height}"}
    for folder in missing:
        if not any(Path(path).stem == folder.name for path in entries) and not (dry_run and folder in found):
            print(f"  WARNING no video found for {folder.name}: training will skip it until it has a video_sets entry.")

    if not entries:
        return
    if dry_run:
        for path, entry in entries.items():
            print(f"  --dry-run, would add: {path}: {entry}")
        return
    shutil.copy2(config, config.with_name(config.name + f".bak-{datetime.now().astimezone():%Y%m%d-%H%M%S}"))
    if cfg.get("video_sets") is None:
        cfg["video_sets"] = {}
    cfg["video_sets"].update(entries)
    with open(config, "w", encoding="utf-8") as handle:
        yaml.dump(cfg, handle)
    for path, entry in entries.items():
        print(f"  added to video_sets: {path}: {entry}")


def read_log(log_path: Path) -> set[str]:
    if not log_path.exists():
        return set()
    with open(log_path, encoding="utf-8") as handle:
        return {row["folder"] for row in csv.DictReader(handle)}


def append_log(log_path: Path, tag: str, folder: Path):
    is_new = not log_path.exists()
    with open(log_path, "a", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        if is_new:
            writer.writerow(["round", "folder", "reviewed_at"])
        writer.writerow([tag, folder.name, datetime.now().isoformat(timespec="seconds")])


def open_single(folder: Path, config: Path, title: str):
    """Child process: one viewer, one folder, exits when the window is closed."""
    warnings.filterwarnings("ignore", category=FutureWarning)
    import napari
    from napari.layers import Points
    from qtpy.QtCore import QTimer

    from napari_deeplabcut.core.layers import set_uniform_point_size

    viewer = napari.Viewer(title=title)
    _, controls = viewer.window.add_plugin_dock_widget("napari-deeplabcut", "Keypoint controls")
    viewer.open(str(folder), plugin="napari-deeplabcut")
    if not any(isinstance(layer, Points) for layer in viewer.layers):
        # Unlabeled folder: the config provides an empty keypoints layer to annotate
        viewer.open(str(config), plugin="napari-deeplabcut")
    start_label = START_LABEL or read_project(config)["bodyparts"][0]

    def apply_start_settings():
        points = [layer for layer in viewer.layers if isinstance(layer, Points)]
        if not points:
            return
        layer = points[0]
        viewer.layers.selection.active = layer
        layer.mode = START_TOOL
        controls.label_mode = LABEL_MODE
        if POINT_SIZE:
            set_uniform_point_size(layer, POINT_SIZE, update_new=True)
        store = controls.layer_manager.get_store(layer)
        if store is not None:
            store.current_label = start_label

    # Once now and once after the plugin's deferred setup, which may switch the layer to "add"
    def select_start_label(event=None):
        store = next((s for _, s in controls.layer_manager.iter_managed_points()), None)
        if store is not None and store.current_label != start_label:
            store.current_label = start_label

    apply_start_settings()
    QTimer.singleShot(500, apply_start_settings)
    if START_LABEL_EACH_FRAME:
        # Connected after the plugin's own frame-change reset, so it runs after it and wins
        QTimer.singleShot(600, lambda: viewer.dims.events.current_step.connect(select_start_label, position="last"))
    napari.run()


def label_mtime(folder: Path, scorer: str) -> float:
    label_file = folder / f"CollectedData_{scorer}.h5"
    return label_file.stat().st_mtime if label_file.exists() else 0.0


def review(config: Path, folders: list[Path], scorer: str, tag: str, include_reviewed: bool):
    log_path = config.parent / "review_log.csv"
    reviewed = read_log(log_path)
    queue = [f for f in folders if include_reviewed or f.name not in reviewed]
    print(f"{len(folders)} folders, {len(folders) - len(queue)} already reviewed, {len(queue)} to go.")
    images = {f: sum(p.suffix.lower() in IMAGE_SUFFIXES for p in f.iterdir()) for f in queue}
    open_seconds = 0.0  # time windows were open this session; the prompts between windows do not count
    frames_done = 0
    rates = []  # seconds per frame of each folder; the median ignores a break taken with a window open
    estimate = ""

    for position, folder in enumerate(queue, start=1):
        folder_seconds = 0.0
        done = len(read_log(log_path) & {f.name for f in folders})
        title = f"Review [{position}/{len(queue)}] (reviewed {done}/{len(folders)}){estimate} - {folder.name}"
        while True:
            print(f"\n[{position}/{len(queue)}] {folder.name} ({images[folder]} frames): correct, Ctrl+S, close the window.")
            before = label_mtime(folder, scorer)
            opened = time.time()
            subprocess.run([sys.executable, __file__, "--single", str(folder), str(config), title], check=False)
            took = time.time() - opened
            open_seconds += took
            folder_seconds += took
            print(f"  window open {format_duration(took)}")
            if label_mtime(folder, scorer) > before:
                append_log(log_path, tag, folder)
                break
            answer = input("Not saved. [r]eopen, mark as [o]k without changes, [s]kip, [q]uit: ").strip().lower()
            if answer == "o":
                append_log(log_path, tag, folder)
                break
            if answer == "s":
                break
            if answer == "q":
                return
        frames_done += images[folder]
        if images[folder]:
            rates.append(folder_seconds / images[folder])
        frames_left = sum(images[f] for f in queue[position:])
        if rates and frames_left:
            per_frame = statistics.median(rates)
            left = per_frame * frames_left
            finish = datetime.now().astimezone() + timedelta(seconds=left)
            estimate = f" ~{format_duration(left)} left, done ~{finish:%H:%M}"
            print(
                f"  session {format_duration(open_seconds)} for {frames_done} frames (median {per_frame:.1f} s/frame) | "
                f"{len(queue) - position} folders, {frames_left} frames left | ~{format_duration(left)} | finish ~{finish:%H:%M}"
            )
    print(f"\n{len(read_log(log_path) & {f.name for f in folders})}/{len(folders)} folders reviewed.")


def format_duration(seconds: float) -> str:
    minutes = int(seconds // 60)
    return f"{minutes // 60}h{minutes % 60:02d}m" if minutes >= 60 else f"{minutes}m{int(seconds % 60):02d}s"


def main():
    if len(sys.argv) >= 2 and sys.argv[1] == "--single":
        open_single(Path(sys.argv[2]), Path(sys.argv[3]), sys.argv[4])
        return

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("config", help="Target project's config.yaml.")
    parser.add_argument("folders", nargs="*", help="Folder names in labeled-data, or paths to import (default: all of labeled-data).")
    parser.add_argument("--list", help="Text file with one folder name or path per line.")
    parser.add_argument("--match", help='Only folders whose name matches this glob, e.g. "TT_*".')
    parser.add_argument("--tag", default="review", help="Label written to review_log.csv for this session.")
    parser.add_argument("--all", action="store_true", help="Also open folders already marked as reviewed.")
    parser.add_argument("--videos-dir", help="Folder that already holds the videos: video_sets points there, no prompt, no copy.")
    parser.add_argument(
        "--dry-run", action="store_true", help="List what would be imported and added to video_sets; change nothing, do not open napari."
    )
    args = parser.parse_args()

    config = Path(args.config)
    project = read_project(config)
    entries = [clean_entry(entry) for entry in args.folders]
    if args.list:
        lines = [clean_entry(line) for line in Path(args.list).read_text(encoding="utf-8").splitlines()]
        entries += [line for line in lines if line and not line.startswith("#")]
    folders, sources = resolve_folders(project, entries, args.match, args.dry_run)
    if not folders:
        raise SystemExit("No folders to review.")
    add_video_sets(config, folders, sources, Path(clean_entry(args.videos_dir)) if args.videos_dir else None, args.dry_run)
    if args.dry_run:
        reviewed = read_log(config.parent / "review_log.csv")
        for folder in folders:
            print(f"  {'[reviewed] ' if folder.name in reviewed else ''}{folder.name}")
        return
    review(config, folders, project["scorer"], args.tag, args.all)


if __name__ == "__main__":
    main()

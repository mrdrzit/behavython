"""
Compares DeepLabCut networks on the same videos, without labels.

Every video directly inside <eval folder> is hard-linked (copied when a link is impossible, e.g. across drives)
into <eval folder>/<Task of each network>/, so each network analyzes its own copy: two networks cannot analyze
the same video in one folder, because behavython skips a video that already has tracking files. Each network
then runs through analyze_dlc_videos.py (behavython-cli tracking, live progress and ETA). The originals in
<eval folder> are only read.

Written to <eval folder>/comparison/:
  analysis_profile.csv and .png    profile_analysis.py over all networks: unsure and jump % per video and bodypart
  hard_stats_<Task>.txt            pick_hard_frames.py statistics per network: unsure % per tenth and odd shape %
  <model>/training_profile.*       profile_training.py per network: test RMSE and mAP per evaluation

  python scripts\\compare_networks.py <eval folder> <config.yaml> <config.yaml> [...]

Without labels this ranks networks by how often they are unsure or produce odd shapes, not by accuracy; a
network can be confidently wrong. Check the frames it flags, or label holdout frames for eval_holdout.py.
"""

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

import yaml

SCRIPTS = Path(__file__).resolve().parent
VIDEO_SUFFIXES = (".mp4", ".avi", ".mov")


def run(args: list, log: Path | None = None) -> None:
    """Runs a script of this folder with the same Python; output goes to the terminal, or to log and the terminal."""
    command = [sys.executable, str(SCRIPTS / args[0]), *map(str, args[1:])]
    print(f"\n=== {' '.join(command[1:])}", flush=True)
    if log is None:
        subprocess.run(command, check=False)
        return
    result = subprocess.run(command, capture_output=True, text=True, encoding="utf-8", errors="replace", check=False)
    log.write_text(result.stdout + result.stderr, encoding="utf-8")
    print(result.stdout + result.stderr, flush=True)


def link_videos(videos: list[Path], target: Path) -> None:
    target.mkdir(exist_ok=True)
    for video in videos:
        destination = target / video.name
        if destination.exists():
            continue
        try:
            os.link(video, destination)
        except OSError:
            shutil.copy2(video, destination)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("eval_folder", help="Folder with the videos to compare on.")
    parser.add_argument("configs", nargs="+", help="config.yaml (or project folder) of each network.")
    args = parser.parse_args()

    eval_folder = Path(args.eval_folder)
    videos = sorted(p for p in eval_folder.iterdir() if p.is_file() and p.suffix.lower() in VIDEO_SUFFIXES)
    if not videos:
        raise SystemExit(f"no videos directly inside {eval_folder}")
    configs = [Path(c) / "config.yaml" if Path(c).is_dir() else Path(c) for c in args.configs]
    networks = {}
    for config in configs:
        with open(config, encoding="utf-8") as handle:
            task = yaml.safe_load(handle)["Task"]
        if task in networks:
            raise SystemExit(f"two networks share the Task {task}: their folders and output names would collide")
        networks[task] = config
    print(f"{len(videos)} videos, {len(networks)} networks: {', '.join(networks)}")

    for task, config in networks.items():
        link_videos(videos, eval_folder / task)
        run(["analyze_dlc_videos.py", eval_folder / task, "--dlc-config", config])

    out = eval_folder / "comparison"
    out.mkdir(exist_ok=True)
    run(["profile_analysis.py", *(eval_folder / task for task in networks), "--out", out])
    for task, config in networks.items():
        # statistics only (no --apply): --out is required but nothing is written there
        run(["pick_hard_frames.py", eval_folder / task, "--out", out / "unused", "--count", "1"], log=out / f"hard_stats_{task}.txt")
        run(["profile_training.py", config.parent, "--out", out])
    print(f"\ncomparison written to {out}")


if __name__ == "__main__":
    main()

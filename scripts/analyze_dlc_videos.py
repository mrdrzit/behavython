"""
Runs DeepLabCut tracking (behavython-cli --run-tracking) on videos and shows timing, progress and an ETA.

Inputs are folders and/or video files. A folder means every .mp4/.avi/.mov directly inside it (--recursive adds
subfolders). Each video runs in its own behavython-cli process, so a video that fails does not stop the others.
The network must be chosen: --network cd1|c57, --dlc-config PATH, or the prompt shown when neither is given.

Before running, a table lists every video with resolution, fps, duration, frames and what will happen to it:
  analyze        no tracking files yet: DLC analysis, then filtering
  filter         .h5 present but no _filtered file: filtering only
  done           tracking and filtered files present: skipped
  other network  tracking files from a different network: skipped (move them away to reanalyze)
  unreadable     OpenCV cannot open the video or read a frame: skipped

While running, one line per video with processing time and speed, and an ETA for the whole batch. The rate is
measured on the finished videos (frames per second of the analysis step, without model loading), so the first
estimate appears after the first analyzed video, or right away with --rate.

--dry-run --probe [CLIPS] measures instead of guessing: it cuts a 1 minute clip (stream copy, original codec) at a
random point between 10% and 90% of CLIPS randomly chosen videos (default 2, different resolutions first), runs the
real behavython-cli tracking on each clip in a temporary folder, and deletes that folder afterwards. The estimate
splits the probe into a fixed cost per video (startup, model loading), the inference speed from DLC's progress bar,
and filtering time per second of video, then applies them to every video to analyze.

Written into each video folder (nothing else is changed by this script):
  dlc_analysis_<timestamp>.log          console output of every behavython-cli run
  dlc_analysis_report_<timestamp>.csv   one row per video, rewritten after every video

  scripts\\analyze_dlc_videos.bat <folders or videos>           launcher: uses the behavython miniforge env
  python scripts\\analyze_dlc_videos.py <paths> --network cd1
  python scripts\\analyze_dlc_videos.py <paths> --dlc-config <config.yaml> --dry-run
  python scripts\\analyze_dlc_videos.py <paths> --network cd1 --dry-run --probe       time 2 clips, then estimate

Needs the behavython repo (this file must stay in its scripts/ folder) and DeepLabCut in the env.
"""

import argparse
import csv
import importlib.metadata
import os
import random
import re
import shutil
import subprocess
import sys
import tempfile
import time
from datetime import datetime, timedelta
from pathlib import Path

import cv2
import yaml

SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC))

from behavython.core.defaults import ANALYSIS_REQUIRED_SUFFIXES, DEFAULT_POSE_NETWORKS
from behavython.core.paths import USER_MODELS_ROOT
from behavython.core.utils import get_ffmpeg_path
from behavython.pipeline.plugins.dlc import get_dlc_file_status

VIDEO_SUFFIXES = tuple(ANALYSIS_REQUIRED_SUFFIXES["video"])
PROBE_SECONDS = 60
PROGRESS_LINE = re.compile(r"Progress: (\d+)%(?: \| ([\d.]+)\s*frames/sec)?")
REPORT_COLUMNS = [
    "video",
    "folder",
    "width",
    "height",
    "fps",
    "duration_s",
    "frames",
    "status",
    "existing_scorer",
    "result",
    "load_s",
    "analysis_s",
    "filter_s",
    "total_s",
    "analysis_fps",
    "error",
    "dlc_config",
]


def format_duration(seconds: float) -> str:
    minutes = int(seconds // 60)
    return f"{minutes // 60}h{minutes % 60:02d}m" if minutes >= 60 else f"{minutes}m{int(seconds % 60):02d}s"


def choose_config(args: argparse.Namespace) -> Path:
    if args.dlc_config:
        path = Path(args.dlc_config)
        return path / "config.yaml" if path.is_dir() else path
    network = args.network
    while network is None:
        names = sorted(DEFAULT_POSE_NETWORKS)
        options = "  ".join(f"[{n}] {name}" for n, name in enumerate(names, 1))
        answer = input(f"Network: {options}  [{len(names) + 1}] path to a config.yaml\n> ").strip().strip('"')
        if answer in [str(n) for n in range(1, len(names) + 1)]:
            network = names[int(answer) - 1]
        elif answer == str(len(names) + 1):
            path = Path(input("config.yaml (or its folder): ").strip().strip('"'))
            return path / "config.yaml" if path.is_dir() else path
        elif answer in names:
            network = answer
    return USER_MODELS_ROOT / DEFAULT_POSE_NETWORKS[network] / "config.yaml"


def collect_videos(paths: list[str], recursive: bool) -> list[Path]:
    videos = []
    for raw in paths:
        path = Path(raw).resolve()
        if path.is_dir():
            files = path.rglob("*") if recursive else path.iterdir()
            # _labeled videos are DLC output, not recordings
            videos += sorted(f for f in files if f.is_file() and f.suffix.lower() in VIDEO_SUFFIXES and not f.stem.endswith("_labeled"))
        elif path.is_file() and path.suffix.lower() in VIDEO_SUFFIXES:
            videos.append(path)
        else:
            print(f"ignored (not a folder or video): {raw}")
    return list(dict.fromkeys(videos))  # dragging a folder and a file inside it lists that video once


def existing_scorer(video: Path) -> str:
    scorers = {f.name[len(video.stem) : -3].removesuffix("_filtered") for f in video.parent.glob(f"{video.stem}DLC*.h5")}
    return ";".join(sorted(scorers))


def inspect_video(video: Path, network_tag: str) -> dict:
    row = {"video": video.name, "folder": str(video.parent), "path": video, "existing_scorer": existing_scorer(video)}
    capture = cv2.VideoCapture(str(video))
    readable = capture.isOpened() and capture.read()[0]
    frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT)) if readable else 0
    fps = capture.get(cv2.CAP_PROP_FPS) if readable else 0.0
    row.update(width=int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)), height=int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    capture.release()
    row.update(frames=frames, fps=round(fps, 2), duration_s=round(frames / fps, 1) if fps else 0.0)
    if frames <= 0:
        row["status"] = "unreadable"
    elif row["existing_scorer"] and network_tag not in row["existing_scorer"]:
        row["status"] = "other network"
    else:
        state = get_dlc_file_status(str(video))
        row["status"] = "analyze" if state["needs_analysis"] else "filter" if state["needs_filtering"] else "done"
    return row


def print_table(rows: list[dict]) -> None:
    folder = None
    for n, row in enumerate(rows, 1):
        if row["folder"] != folder:
            folder = row["folder"]
            print(f"\n{folder}")
            print(f"  {'#':>3}  {'video':<32} {'res':>9} {'fps':>6} {'duration':>9} {'frames':>8}  status")
        res = f"{row['width']}x{row['height']}"
        scorer = f"  ({row['existing_scorer']})" if row["status"] == "other network" else ""
        print(
            f"  {n:>3}  {row['video'][:32]:<32} {res:>9} {row['fps']:>6} {format_duration(row['duration_s']):>9} "
            f"{row['frames']:>8}  {row['status']}{scorer}"
        )


def gpu_name() -> str:
    if not shutil.which("nvidia-smi"):
        return "no NVIDIA GPU found: DeepLabCut will run on the CPU (much slower)"
    result = subprocess.run(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"], capture_output=True, text=True, check=False)
    return result.stdout.strip().replace("\n", ", ") or "nvidia-smi found no GPU"


def write_report(path: Path, rows: list[dict], config: Path) -> None:
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, REPORT_COLUMNS, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({**row, "dlc_config": str(config)})


def live(text: str) -> None:
    print(f"\r{text[:110]:<110}", end="", flush=True)


def run_video(row: dict, config: Path, log) -> None:
    """Runs behavython-cli on one video, prints live progress, and stores timings and result in row."""
    env = {**os.environ, "PYTHONPATH": os.pathsep.join([str(SRC), os.environ.get("PYTHONPATH", "")]), "PYTHONUNBUFFERED": "1"}
    env["PYTHONIOENCODING"] = "utf-8"
    command = [sys.executable, "-m", "behavython.cli", "--input-files", str(row["path"]), "--run-tracking", "--dlc-config", str(config)]
    log.write(f"\n=== {datetime.now():%Y-%m-%d %H:%M:%S} {row['video']}\n")  # noqa: DTZ005
    log.flush()
    start = time.time()
    analysis_start = filter_start = None
    error = ""
    speeds = []
    live(f"  {row['video']}: loading DeepLabCut")
    process = subprocess.Popen(
        command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, env=env, text=True, encoding="utf-8", errors="replace"
    )
    try:
        for line in process.stdout:
            log.write(line)
            if "Processing video" in line:
                analysis_start = time.time()
                live(f"  {row['video']}: analyzing")
            elif "Filtering predictions" in line:
                filter_start = time.time()
                live(f"  {row['video']}: filtering")
            elif match := PROGRESS_LINE.search(line):
                percent, speed = match.groups()
                text = f"  {row['video']}: analyzing {percent}%"
                if speed and float(speed) > 0:
                    speeds.append(float(speed))
                    left = row["frames"] * (100 - int(percent)) / 100 / float(speed)
                    text += f" | {float(speed):.0f} fps | video left ~{format_duration(left)}"
                live(text)
            elif "| ERROR" in line:
                error = line.split("| ERROR", 1)[1].strip(" |\n")
        process.wait()
    except KeyboardInterrupt:
        process.terminate()
        process.wait()
        row.update(result="stopped", error="stopped by user")
        raise
    finally:
        end = time.time()
        log.flush()
        print()
        loaded = analysis_start or filter_start or end
        row["load_s"] = round(loaded - start, 1)
        if analysis_start:
            row["analysis_s"] = round((filter_start or end) - analysis_start, 1)
            row["analysis_fps"] = round(row["frames"] / row["analysis_s"], 1) if row["analysis_s"] else ""
        if filter_start:
            row["filter_s"] = round(end - filter_start, 1)
        row["total_s"] = round(end - start, 1)
        if speeds:
            row["inference_fps"] = round(sum(speeds) / len(speeds), 1)

    if not get_dlc_file_status(str(row["path"]))["needs_filtering"]:
        row["result"] = "ok"
    else:
        row["result"] = "failed"
        row["error"] = error or f"exit code {process.returncode}, no filtered output (see the log)"
        row.pop("analysis_fps", None)  # a run that stopped early has no meaningful speed


def estimate_left(rows: list[dict], todo: list[dict], rate: float | None) -> float | None:
    """Seconds left for the videos in todo: frames at the measured analysis rate plus the mean per-video overhead."""
    analyzed = [r for r in rows if r.get("result") == "ok" and r.get("analysis_s")]
    if analyzed:
        rate = sum(r["frames"] for r in analyzed) / sum(r["analysis_s"] for r in analyzed)
    if not rate:
        return None
    finished = [r for r in rows if r.get("result") == "ok"]
    overhead = sum(r["load_s"] + r.get("filter_s", 0) for r in finished) / len(finished) if finished else 0
    frames = sum(r["frames"] for r in todo if r["status"] == "analyze")
    return frames / rate + overhead * len(todo)


def pick_probe_videos(candidates: list[dict], count: int) -> list[dict]:
    """Random videos, taking a resolution not picked yet whenever one is left."""
    pool = random.sample(candidates, len(candidates))
    picks = []
    while pool and len(picks) < count:
        seen = {(r["width"], r["height"]) for r in picks}
        row = next((r for r in pool if (r["width"], r["height"]) not in seen), pool[0])
        pool.remove(row)
        picks.append(row)
    return picks


def run_probe(picks: list[dict], config: Path, network_tag: str) -> list[dict]:
    """Cuts a clip of each picked video into a temporary folder, tracks it, and returns the measured clip rows."""
    ffmpeg = get_ffmpeg_path()
    clips = []
    with tempfile.TemporaryDirectory(prefix="behavython_probe_", ignore_cleanup_errors=True) as tmp:  # noqa: SIM117  (log needs tmp)
        with open(Path(tmp) / "probe.log", "a", encoding="utf-8") as log:
            for n, row in enumerate(picks, 1):
                folder = Path(tmp) / str(n)
                folder.mkdir()
                clip = folder / row["path"].name
                duration = row["duration_s"]
                if duration <= PROBE_SECONDS:
                    start = 0.0
                    shutil.copy2(row["path"], clip)
                else:
                    start = random.uniform(0.1 * duration, max(0.1 * duration, 0.9 * duration - PROBE_SECONDS))
                    command = [ffmpeg, "-v", "error", "-ss", f"{start:.1f}", "-i", str(row["path"])]
                    command += ["-t", str(PROBE_SECONDS), "-c", "copy", "-an", str(clip)]
                    subprocess.run(command, check=True)
                clip_row = inspect_video(clip, network_tag)
                print(f"probe [{n}/{len(picks)}] {row['path']} from {format_duration(start)}: {clip_row['frames']} frames")
                if clip_row["status"] != "analyze":
                    print(f"  clip is {clip_row['status']}, skipped")
                    continue
                run_video(clip_row, config, log)
                if clip_row["result"] != "ok":
                    print(f"  probe failed: {clip_row['error']}")
                    continue
                inference = clip_row.get("inference_fps") or clip_row["analysis_fps"]
                # model building inside analyze_videos: analysis time not explained by the frames
                clip_row["init_s"] = max(0.0, clip_row["analysis_s"] - clip_row["frames"] / inference)
                clip_row["inference"] = inference
                print(
                    f"  {row['width']}x{row['height']}: load {clip_row['load_s']:.0f} s + model {clip_row['init_s']:.0f} s | "
                    f"inference {inference:.0f} fps | filter {clip_row.get('filter_s', 0):.1f} s"
                )
                clips.append(clip_row)
    return clips


def probe_estimate(clips: list[dict], todo: list[dict]) -> tuple[float, float, float]:
    """Seconds for todo from the probe: (mean, fastest, slowest) inference speed with the mean fixed costs."""
    load = sum(c["load_s"] for c in clips) / len(clips)
    init = sum(c["init_s"] for c in clips) / len(clips)
    filter_per_s = sum(c.get("filter_s", 0) / c["duration_s"] for c in clips) / len(clips)

    def total(inference: float) -> float:
        seconds = 0.0
        for row in todo:
            seconds += load + filter_per_s * row["duration_s"]
            if row["status"] == "analyze":
                seconds += init + row["frames"] / inference
        return seconds

    speeds = [c["inference"] for c in clips]
    return total(sum(speeds) / len(speeds)), total(max(speeds)), total(min(speeds))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("paths", nargs="+", help="Folders and/or video files.")
    network = parser.add_mutually_exclusive_group()
    network.add_argument("--network", choices=sorted(DEFAULT_POSE_NETWORKS), help="Downloaded behavython network.")
    network.add_argument("--dlc-config", metavar="PATH", help="config.yaml (or its folder) of a DeepLabCut project.")
    parser.add_argument("--recursive", action="store_true", help="Also take videos in subfolders of the given folders.")
    parser.add_argument("--dry-run", action="store_true", help="Show the table and the totals, run nothing, write nothing.")
    parser.add_argument(
        "--rate", type=float, metavar="FPS", help="Expected analysis frames per second, for an ETA before the first video ends."
    )
    parser.add_argument(
        "--probe",
        nargs="?",
        const=2,
        type=int,
        metavar="CLIPS",
        help=f"With --dry-run: track {PROBE_SECONDS} s clips of CLIPS random videos (default 2) in a temporary folder and estimate from them.",
    )
    args = parser.parse_args()
    if args.probe is not None and (not args.dry_run or args.probe < 1):
        parser.error("--probe needs --dry-run and at least 1 clip; a real run measures its first videos anyway")

    videos = collect_videos(args.paths, args.recursive)
    if not videos:
        raise SystemExit("no videos found")
    config = choose_config(args)
    if not config.is_file():
        raise SystemExit(f"config.yaml not found: {config}")
    with open(config, encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle)
    network_tag = f"{cfg['Task']}{cfg['date']}"  # part of the scorer name in DLC output files

    rows = [inspect_video(video, network_tag) for video in videos]
    print_table(rows)
    todo = [r for r in rows if r["status"] in ("analyze", "filter")]
    to_analyze = [r for r in todo if r["status"] == "analyze"]
    try:
        dlc_version = importlib.metadata.version("deeplabcut")
    except importlib.metadata.PackageNotFoundError:
        dlc_version = "not installed"
    print(f"\nnetwork: {config} (Task {cfg['Task']}, date {cfg['date']}, snapshotindex {cfg.get('snapshotindex')})")
    print(f"DeepLabCut {dlc_version} | GPU: {gpu_name()}")
    print(
        f"to analyze: {len(to_analyze)} videos, {format_duration(sum(r['duration_s'] for r in to_analyze))} of video, "
        f"{sum(r['frames'] for r in to_analyze)} frames | filter only: {len(todo) - len(to_analyze)} | "
        f"skipped: {len(rows) - len(todo)}"
    )
    left = estimate_left(rows, todo, args.rate)
    if left is not None:
        print(f"estimate: ~{format_duration(left)} at {args.rate} fps, without model loading")
    if args.probe and to_analyze:
        picks = pick_probe_videos(to_analyze, args.probe)
        sizes = {(r["width"], r["height"]) for r in to_analyze} - {(r["width"], r["height"]) for r in picks}
        if sizes:
            print(f"warning: resolutions not probed: {', '.join(f'{w}x{h}' for w, h in sorted(sizes))}")
        print()
        try:
            clips = run_probe(picks, config, network_tag)
        except KeyboardInterrupt:
            print("probe stopped by user")
            return
        except (RuntimeError, subprocess.CalledProcessError) as exc:  # ffmpeg missing or failed to cut
            raise SystemExit(f"probe failed: {exc}") from exc
        if clips:
            mean, fast, slow = probe_estimate(clips, todo)
            print(
                f"estimate: ~{format_duration(mean)} for {len(todo)} videos (range {format_duration(fast)} to "
                f"{format_duration(slow)}) | ETA {datetime.now() + timedelta(seconds=mean):%H:%M} if started now"  # noqa: DTZ005
            )
        else:
            print("no probe clip finished: no estimate")
    if args.dry_run or not todo:
        print("dry run: nothing was run" if args.dry_run else "nothing to run")
        return

    stamp = f"{datetime.now():%Y-%m-%d_%H-%M-%S}"  # noqa: DTZ005
    folders = list(dict.fromkeys(r["folder"] for r in todo))  # folders with nothing to run get no report
    reports = {folder: Path(folder) / f"dlc_analysis_report_{stamp}.csv" for folder in folders}
    for row in rows:
        if row not in todo:
            row["result"] = "skipped"
    started = time.time()
    try:
        for n, row in enumerate(todo, 1):
            with open(Path(row["folder"]) / f"dlc_analysis_{stamp}.log", "a", encoding="utf-8") as log:
                print(f"[{n}/{len(todo)}] {row['video']} ({row['status']}, {format_duration(row['duration_s'])}, {row['frames']} frames)")
                run_video(row, config, log)
            folder_rows = [r for r in rows if r["folder"] == row["folder"]]
            write_report(reports[row["folder"]], folder_rows, config)
            parts = [f"[{n}/{len(todo)}] {row['video']} {row['result']} in {format_duration(row['total_s'])}"]
            if row.get("analysis_fps"):
                parts.append(f"analysis {format_duration(row['analysis_s'])} ({row['analysis_fps']:.0f} fps), load {row['load_s']:.0f} s")
            if row["result"] == "failed":
                parts.append(row["error"][:80])
            left = estimate_left(rows, todo[n:], args.rate)
            if n < len(todo) and left is not None:
                parts.append(f"left ~{format_duration(left)} | ETA {datetime.now() + timedelta(seconds=left):%H:%M}")  # noqa: DTZ005
            print(" | ".join(parts), flush=True)
    except KeyboardInterrupt:
        print("stopped by user")
    finally:
        for folder in folders:
            write_report(reports[folder], [r for r in rows if r["folder"] == folder], config)

    ok = [r for r in todo if r.get("result") == "ok"]
    failed = [r for r in todo if r.get("result") == "failed"]
    print(f"\nfinished in {format_duration(time.time() - started)}: {len(ok)} ok, {len(failed)} failed, {len(rows) - len(todo)} skipped")
    analyzed = [r for r in ok if r.get("analysis_s")]
    if analyzed:
        rate = sum(r["frames"] for r in analyzed) / sum(r["analysis_s"] for r in analyzed)
        slowest = min(analyzed, key=lambda r: r["analysis_fps"])
        print(f"mean analysis speed {rate:.0f} fps | slowest {slowest['video']} ({slowest['analysis_fps']:.0f} fps)")
    for row in failed:
        print(f"failed: {row['path']}: {row['error']}")
    for report in reports.values():
        print(f"report: {report}")


if __name__ == "__main__":
    main()

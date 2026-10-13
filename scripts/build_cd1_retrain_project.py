"""
Builds a fresh DLC 3 (PyTorch) project for the CD1 retraining.

build   : copies every labeled folder from the base project, selects new videos
          (holdout / bootstrap / outlier) from the behavior dataset, stages them
          as hard links inside the new project and writes config.yaml + manifest.csv.
frames  : picks frames for every staged video from its existing Feb13 predictions
          (skipping the first 10% of each video) and writes them to frames/<round>/<stem>/,
          ready for the Behavython assisted-label tool.
collect : moves the folders labeled by the Behavython assisted-label tool from
          frames/round1, round2 and holdout into labeled-data (never overwrites).

Source videos are never moved, copied into, or written next to.
"""

import argparse
import csv
import math
import os
import random
import re
import shutil
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from datetime import date
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from deeplabcut.utils import auxiliaryfunctions

VIDEO_ROOT = Path(r"E:\Gustavo\Comportamentos")
NETWORKS_ROOT = Path(r"E:\Matheus\Deeplabcut_networks")
BASE_PROJECT = NETWORKS_ROOT / "CD1-GUSTAVO-2026-09-18"
TASK, SCORER = "CD1", "GUSTAVO"

# Folders holding exactly one copy of each video (the "All*" / "*rede_nova" folders are duplicates)
CANONICAL_DIRS = {"D1_Treino", "D2_Teste", "Dia_1_Treino", "Dia_2_Teste", "All_levas_3e4"}
STEM_PATTERN = re.compile(r"^(?P<phase>T[RT])_A(?P<animal>\d+)_(?:20)?(?P<date>\d{2}-\d{2}-\d{2})_(?P<time>\d{2}-\d{2}-\d{2})$")

AGRUP_CA2 = "pos-pintura_AGRUP_CD1_OXT_ANTAG_CA2_PRETT"
ISO_CA2 = "pos-pintura_ISO_CD1_OXT_ANTAG_CA2_PRETT"
RETESTE = "pos-pintura_RETESTE_GReSI_antag_VTA_062026"
BASAL_AGRUP = "pos-pintura_BASAL_AGRUP_CD1"
BASAL_ISO = "pos-pintura_BASAL_ISO_CD1"
PRE_ANTAG = "pre-pintura_AGRUP_RS_CD1_OXT_ANTAG_VTA_PRETT"
PRE_PRETR = "pre-pintura_AGRUP_RS_CD1_OXT_PRETR_VTA_12_2025"

# Holdout: one never-trained video per experiment, drawn before anything else
HOLDOUT_GROUPS = [AGRUP_CA2, ISO_CA2, RETESTE, BASAL_AGRUP, BASAL_ISO, PRE_ANTAG, PRE_PRETR]

# (experiment, batch, day) -> number of videos
QUOTAS = {
    "bootstrap_pos": {
        (AGRUP_CA2, "Leva_2", "D1"): 4,
        (AGRUP_CA2, "Leva_2", "D2"): 4,
        (AGRUP_CA2, "Leva_1", "D1"): 2,
        (AGRUP_CA2, "Leva_1", "D2"): 3,
        (ISO_CA2, "", "D1"): 2,
        (ISO_CA2, "", "D2"): 4,
        (RETESTE, "", "D1"): 2,
        (RETESTE, "", "D2"): 4,
        (BASAL_AGRUP, "", "D1"): 2,
        (BASAL_AGRUP, "", "D2"): 1,
        (BASAL_ISO, "", "D1"): 2,
        (BASAL_ISO, "", "D2"): 1,
    },
    "outliers": {
        (AGRUP_CA2, "Leva_2", "D1"): 2,
        (AGRUP_CA2, "Leva_2", "D2"): 1,
        (AGRUP_CA2, "Leva_1", "D1"): 1,
        (AGRUP_CA2, "Leva_1", "D2"): 1,
        (ISO_CA2, "", "D1"): 1,
        (ISO_CA2, "", "D2"): 2,
        (RETESTE, "", "D1"): 1,
        (RETESTE, "", "D2"): 1,
        (BASAL_AGRUP, "", "D1"): 1,
        (BASAL_ISO, "", "D2"): 1,
    },
    "bootstrap_pre": {
        (PRE_ANTAG, "Leva_1", "D1"): 1,
        (PRE_ANTAG, "Leva_1", "D2"): 1,
        (PRE_ANTAG, "Leva_2", "D1"): 1,
        (PRE_ANTAG, "Leva_2", "D2"): 1,
        (PRE_ANTAG, "Levas_3e4", "D1"): 1,
        (PRE_ANTAG, "Levas_3e4", "D2"): 1,
        (PRE_PRETR, "Leva_1", "D1"): 1,
        (PRE_PRETR, "Leva_1", "D2"): 1,
        (PRE_PRETR, "Leva_2", "D1"): 1,
        (PRE_PRETR, "Leva_2", "D2"): 1,
    },
}
STAGED_ROLES = ["holdout", "bootstrap_pos", "bootstrap_pre", "outliers"]
# Holdout folders can live in labeled-data too: their videos are not in video_sets, so training ignores them
COLLECTED_ROUNDS = ["round1", "round2", "holdout"]

# Frame picking (frames subcommand)
PREDICTION_MODEL = "DLC_resnet50_CD1_GUSTAVOFeb13shuffle1_950000"  # every staged video was analyzed with it
SKIP_START_FRACTION = 0.10  # experimenter placing the animal in the arena
MIN_GAP_FRAMES = 30  # at least 1 s between frames of the same video
LOW_LIKELIHOOD = 0.6  # pcutoff
HIGH_LIKELIHOOD = 0.9
JUMP_PIXELS = 20  # same epsilon as DLC's "jump" outlier algorithm
# role -> (confident frames, hard frames, uniform frames) per video
FRAMES_PER_VIDEO = {
    "bootstrap_pos": (8, 4, 0),
    "bootstrap_pre": (7, 3, 0),
    "outliers": (0, 20, 0),
    "holdout": (0, 0, 20),  # unbiased sample for evaluation
}


def inventory_videos() -> dict[str, dict]:
    """One record per video stem, taken from the canonical folders only."""
    videos = {}
    for path in VIDEO_ROOT.rglob("*.mp4"):
        relative = path.relative_to(VIDEO_ROOT).parts
        if relative[0].startswith("nao-usar") or path.parent.name not in CANONICAL_DIRS:
            continue
        match = STEM_PATTERN.match(path.stem)
        if not match:
            continue
        batch = next((p for p in relative[1:-1] if p.startswith("Leva")), "")
        videos[path.stem] = {
            "stem": path.stem,
            "path": path,
            "group": relative[0],
            "batch": batch,
            "day": "D1" if match["phase"] == "TR" else "D2",
            "animal": match["animal"],
            "timestamp": f"{match['date']}_{match['time']}",
        }
    return videos


def pick_spread(candidates: list[dict], count: int, rng: random.Random) -> list[dict]:
    """Picks one video per equal-sized time bin so selections span dates, times of day and animals."""
    if count > len(candidates):
        raise SystemExit(f"Not enough candidates ({len(candidates)}) for quota {count}: {candidates[0]['group'] if candidates else '?'}")
    ordered = sorted(candidates, key=lambda v: v["timestamp"])
    bins = [ordered[round(i * len(ordered) / count) : round((i + 1) * len(ordered) / count)] for i in range(count)]
    return [rng.choice(b) for b in bins]


def select_videos(videos: dict[str, dict], labeled: set[str], seed: int) -> dict[str, list[dict]]:
    rng = random.Random(seed)
    available = {stem: v for stem, v in videos.items() if stem not in labeled}
    selection = defaultdict(list)

    def take(role: str, chosen: list[dict]):
        for video in chosen:
            available.pop(video["stem"])
            selection[role].append(video)

    for group in HOLDOUT_GROUPS:
        take("holdout", pick_spread([v for v in available.values() if v["group"] == group], 1, rng))

    for role, quotas in QUOTAS.items():
        for (group, batch, day), count in quotas.items():
            cell = [v for v in available.values() if v["group"] == group and v["batch"] == batch and v["day"] == day]
            take(role, pick_spread(cell, count, rng))
    return selection


def frame_size(path: Path) -> tuple[int, int]:
    capture = cv2.VideoCapture(str(path))
    try:
        width, height = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)), int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    finally:
        capture.release()
    if not width or not height:
        raise SystemExit(f"Could not read frame size of {path}")
    return width, height


def link_or_copy(source: Path, destination: Path):
    """Hard link (same volume, no extra space, no admin rights); falls back to a copy."""
    if destination.exists():
        return
    try:
        os.link(source, destination)
    except OSError:
        shutil.copy2(source, destination)


def build(args):
    project = NETWORKS_ROOT / f"{TASK}-{SCORER}-{args.date}"
    if project.exists():
        raise SystemExit(f"{project} already exists; refusing to touch it.")

    base_labeled = BASE_PROJECT / "labeled-data"
    labeled = sorted(p.name for p in base_labeled.iterdir() if (p / f"CollectedData_{SCORER}.h5").exists())
    videos = inventory_videos()
    for stem in [s for s in labeled if s not in videos]:
        # A few canonical copies were renamed (e.g. "<stem> (1).mp4"); use any exact-name copy instead
        fallback = next((p for p in VIDEO_ROOT.rglob(f"{stem}.mp4") if not p.relative_to(VIDEO_ROOT).parts[0].startswith("nao-usar")), None)
        if fallback:
            match = STEM_PATTERN.match(stem)
            videos[stem] = {
                "stem": stem,
                "path": fallback,
                "group": fallback.relative_to(VIDEO_ROOT).parts[0],
                "batch": "",
                "day": "D1" if match["phase"] == "TR" else "D2",
                "animal": match["animal"],
                "timestamp": "",
            }
    missing = [stem for stem in labeled if stem not in videos]
    if missing:
        raise SystemExit(f"Labeled folders without a source video: {missing}")

    selection = select_videos(videos, set(labeled), args.seed)
    selection["existing"] = [videos[stem] for stem in labeled]

    print(f"Project: {project}")
    for role in ["existing", *STAGED_ROLES]:
        print(f"  {role:<14} {len(selection[role]):>3} videos")
    if not args.apply:
        for role in STAGED_ROLES:
            print(f"\n[{role}]")
            for v in selection[role]:
                print(f"  {v['group']}/{v['batch'] or '-'}/{v['day']}  {v['stem']}")
        print("\nDry run. Re-run with --apply to create the project.")
        return

    for sub in ["labeled-data", "videos", "training-datasets", "dlc-models-pytorch", "analysis"]:
        (project / sub).mkdir(parents=True)

    for stem in labeled:
        destination = project / "labeled-data" / stem
        destination.mkdir()
        for item in (base_labeled / stem).iterdir():
            if item.suffix.lower() == ".png" or item.name.startswith(f"CollectedData_{SCORER}."):
                shutil.copy2(item, destination / item.name)

    rows, video_sets = [], {}
    for role in ["existing", *STAGED_ROLES]:
        for v in selection[role]:
            width, height = frame_size(v["path"])
            if role != "holdout":
                # The key's stem is what create_training_dataset matches against labeled-data/<stem>
                video_sets[str(v["path"])] = {"crop": f"0, {width}, 0, {height}"}
            if role in STAGED_ROLES:
                staging = project / "staging" / role
                staging.mkdir(parents=True, exist_ok=True)
                link_or_copy(v["path"], staging / v["path"].name)
            rows.append(
                {
                    "role": role,
                    **{k: v[k] for k in ("stem", "group", "batch", "day", "animal")},
                    "width": width,
                    "height": height,
                    "source": v["path"],
                }
            )

    cfg = auxiliaryfunctions.read_config(str(BASE_PROJECT / "config.yaml"))
    cfg.update(
        project_path=str(project),
        date=date.fromisoformat(args.date).strftime("%b%d"),
        iteration=0,
        snapshotindex=-1,
        video_sets=video_sets,
    )
    auxiliaryfunctions.write_config(str(project / "config.yaml"), cfg)

    with open(project / "manifest.csv", "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Created {project} ({len(video_sets)} video_sets entries, {len(labeled)} labeled folders copied).")


def find_predictions(stems: set[str]) -> dict[str, Path]:
    """Unfiltered Feb13 predictions (likelihoods are what matters, so filtering is irrelevant)."""
    found = {}
    for h5 in VIDEO_ROOT.rglob(f"*{PREDICTION_MODEL}.h5"):
        stem = h5.name[: -len(f"{PREDICTION_MODEL}.h5")]
        if stem in stems:
            found.setdefault(stem, h5)
    return found


def classify_frames(h5: Path) -> tuple[int, np.ndarray, np.ndarray]:
    """Returns (n_frames, min likelihood per frame, max jump in pixels per frame)."""
    df = pd.read_hdf(h5)
    likelihood = df.xs("likelihood", level="coords", axis=1).to_numpy()
    x = df.xs("x", level="coords", axis=1).to_numpy()
    y = df.xs("y", level="coords", axis=1).to_numpy()
    jump = np.zeros(len(df))
    jump[1:] = np.nanmax(np.hypot(np.diff(x, axis=0), np.diff(y, axis=0)), axis=1)
    return len(df), likelihood.min(axis=1), jump


def spread_pick(candidates: np.ndarray, count: int, taken: list[int], rng: random.Random) -> list[int]:
    """Picks `count` frames spread over the candidates' time span, keeping MIN_GAP_FRAMES between all picks."""
    picked = []
    if count == 0 or len(candidates) == 0:
        return picked
    chunks = np.array_split(np.sort(candidates), min(count, len(candidates)))
    for chunk in chunks:
        options = [int(f) for f in chunk if all(abs(int(f) - t) >= MIN_GAP_FRAMES for t in taken + picked)]
        if options:
            picked.append(rng.choice(options))
    return picked


def pick_frames(n_frames: int, min_likelihood: np.ndarray, jump: np.ndarray, role: str, rng: random.Random) -> list[tuple[int, str]]:
    confident_n, hard_n, uniform_n = FRAMES_PER_VIDEO[role]
    frames = np.arange(int(n_frames * SKIP_START_FRACTION), n_frames)
    hard = frames[(min_likelihood[frames] < LOW_LIKELIHOOD) | (jump[frames] > JUMP_PIXELS)]
    confident = frames[(min_likelihood[frames] >= HIGH_LIKELIHOOD) & (jump[frames] <= JUMP_PIXELS)]

    taken: list[tuple[int, str]] = []
    for kind, pool, count in [("hard", hard, hard_n), ("confident", confident, confident_n), ("uniform", frames, uniform_n)]:
        chosen = spread_pick(pool, count, [f for f, _ in taken], rng)
        if len(chosen) < count:
            # Too few frames of this kind (e.g. a video Feb13 handled well): top up from the whole usable range
            chosen += spread_pick(frames, count - len(chosen), [f for f, _ in taken] + chosen, rng)
        taken += [(f, kind) for f in chosen]
    return sorted(taken)


def write_frames(source: str, destination: str, frames: list[int], n_frames: int) -> list[int]:
    """Decodes sequentially (exactly how DLC indexed the predictions) and saves the requested frames."""
    destination = Path(destination)
    destination.mkdir(parents=True)
    digits = int(math.ceil(math.log10(n_frames)))  # same naming as deeplabcut.extract_frames
    wanted, written = set(frames), []
    capture = cv2.VideoCapture(source)
    try:
        for index in range(max(frames) + 1):
            if not capture.grab():
                break
            if index in wanted:
                ok, image = capture.retrieve()
                if ok and cv2.imwrite(str(destination / f"img{index:0{digits}d}.png"), image):
                    written.append(index)
    finally:
        capture.release()
    return written


def frames(args):
    project = Path(args.project)
    output = project / "frames"
    if output.exists():
        raise SystemExit(f"{output} already exists; delete it to pick frames again.")

    rows = [r for r in csv.DictReader(open(project / "manifest.csv", encoding="utf-8")) if r["role"] in FRAMES_PER_VIDEO]
    predictions = find_predictions({r["stem"] for r in rows})
    missing = [r["stem"] for r in rows if r["stem"] not in predictions]
    if missing:
        raise SystemExit(f"No {PREDICTION_MODEL} predictions for: {missing}")

    # Round 1: all pre-pintura + every other pos-pintura video; round 2: the rest of pos + outliers
    pos = sorted((r for r in rows if r["role"] == "bootstrap_pos"), key=lambda r: (r["group"], r["batch"], r["day"], r["stem"]))
    round_of = {r["stem"]: "round1" for r in rows if r["role"] == "bootstrap_pre"}
    round_of.update({r["stem"]: ("round1" if i % 2 == 0 else "round2") for i, r in enumerate(pos)})
    round_of.update({r["stem"]: "round2" for r in rows if r["role"] == "outliers"})
    round_of.update({r["stem"]: "holdout" for r in rows if r["role"] == "holdout"})

    rng = random.Random(args.seed)
    plan = []
    for row in rows:
        n_frames, min_likelihood, jump = classify_frames(predictions[row["stem"]])
        picks = pick_frames(n_frames, min_likelihood, jump, row["role"], rng)
        plan.append((row, n_frames, picks, min_likelihood))

    summary = defaultdict(lambda: defaultdict(int))
    for row, _, picks, _ in plan:
        for _, kind in picks:
            summary[round_of[row["stem"]]][kind] += 1
    for name in ["round1", "round2", "holdout"]:
        videos = sum(1 for r, *_ in plan if round_of[r["stem"]] == name)
        print(f"  {name:<8} {videos:>3} videos  " + "  ".join(f"{k}={v}" for k, v in sorted(summary[name].items())))
    if not args.apply:
        print("\nDry run. Re-run with --apply to extract the frames.")
        return

    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        jobs = {
            pool.submit(
                write_frames, row["source"], str(output / round_of[row["stem"]] / row["stem"]), [f for f, _ in picks], n_frames
            ): row["stem"]
            for row, n_frames, picks, _ in plan
        }
        written = {jobs[job]: set(job.result()) for job in jobs}

    with open(output / "frames_manifest.csv", "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["round", "role", "stem", "frame", "kind", "feb13_min_likelihood", "written"])
        for row, _, picks, min_likelihood in plan:
            for frame, kind in picks:
                writer.writerow(
                    [
                        round_of[row["stem"]],
                        row["role"],
                        row["stem"],
                        frame,
                        kind,
                        f"{min_likelihood[frame]:.3f}",
                        frame in written[row["stem"]],
                    ]
                )
    failed = sum(len(picks) - len(written[r["stem"]]) for r, _, picks, _ in plan)
    print(f"Frames written to {output}" + (f" ({failed} frames could not be decoded, see frames_manifest.csv)" if failed else ""))


def collect(args):
    project = Path(args.project)
    labeled_data = project / "labeled-data"
    for role in COLLECTED_ROUNDS:
        staging = project / "frames" / role
        if not staging.exists():
            continue
        for folder in sorted(p for p in staging.iterdir() if p.is_dir() and p.name not in ("temp_folder", "verification_plots")):
            if not (folder / f"CollectedData_{SCORER}.h5").exists():
                print(f"skip (no labels yet): {folder.name}")
            elif (labeled_data / folder.name).exists():
                print(f"skip (already in labeled-data): {folder.name}")
            else:
                shutil.move(str(folder), str(labeled_data / folder.name))
                print(f"moved: {role}/{folder.name}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    commands = parser.add_subparsers(dest="command", required=True)

    build_parser = commands.add_parser("build", help="Create the new project (dry run unless --apply).")
    build_parser.add_argument("--date", default=date.today().isoformat(), help="Project date, YYYY-MM-DD (default: today).")
    build_parser.add_argument("--seed", type=int, default=20261012, help="Random seed for the video selection.")
    build_parser.add_argument("--apply", action="store_true", help="Actually create the project.")
    build_parser.set_defaults(func=build)

    frames_parser = commands.add_parser("frames", help="Pick and extract frames for the staged videos (dry run unless --apply).")
    frames_parser.add_argument("project", help="Path to the new project folder.")
    frames_parser.add_argument("--seed", type=int, default=20261012, help="Random seed for the frame selection.")
    frames_parser.add_argument("--workers", type=int, default=4, help="Videos decoded in parallel.")
    frames_parser.add_argument("--apply", action="store_true", help="Actually extract the frames.")
    frames_parser.set_defaults(func=frames)

    collect_parser = commands.add_parser("collect", help="Move pre-labeled folders from frames/<round> into labeled-data.")
    collect_parser.add_argument("project", help="Path to the new project folder.")
    collect_parser.set_defaults(func=collect)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()

"""
Picks the hardest frames of analyzed videos, to label them for a test set or to add them to a network's training set.

Inputs are folders as behavython leaves them after tracking: each video next to its DLC .h5 files. A frame is hard when
  unsure      the network gives a low likelihood to one of its bodyparts
  shape       the distances between bodyparts are unusual for this animal (a point on a reflection, swapped ears,
              an odd pose); likelihood alone misses confident errors with a BCE head
  disagree    with --compare: a second network puts a bodypart elsewhere (for test sets: pre-label and score with
              two networks, so neither gets an easier test, see AGENTS.md)
Frames come only from the --start..--end part of each video (default: skip the first 10%, where the animal is
placed), are at least --min-gap frames apart, and are skipped when the network does not see the animal.

  python scripts\\pick_hard_frames.py <folders...> --out DIR [--project config.yaml] [--compare DIR ...] [--apply]

Without --apply it only prints statistics per video and writes nothing. With it, for every video:
  <out>/<video stem>/imgNNNNN.png        the picked frames (NNNNN = frame number, padded like DLC does)
  <out>/<video stem>/CollectedData_<scorer>.h5/.csv   the network's predictions on those frames, as pre-labels
  <out>/hard_manifest.csv                video, frame, score and its parts, for every picked frame
Existing frames in <out> are kept; only new ones are added. Videos and tracking files are only read.

--project <config.yaml> skips frames already labeled in that project's labeled-data/<video stem> and writes the
pre-labels with its scorer and bodyparts. Import the result with dlc_review.py <config.yaml> <out>/<stem> ...

Older form (two analysis folders for a test set):
  pick_hard_frames.py <videos> <analysis_A> <analysis_B> <out>   ->   pick_hard_frames.py <videos> --scorer <A>
  --compare <analysis_B> --out <out>, with the h5 files of A next to the videos.
"""

import argparse
import csv
import re
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

VIDEO_SUFFIXES = (".mp4", ".avi", ".mov")
DISAGREE_PX = 20.0  # distance between the two networks that counts as fully hard
SHAPE_Z = 10.0  # robust z of a bodypart-pair distance that counts as fully hard
LOST = 0.1  # pre-label points below this likelihood are left empty (the network lost them)


def find_h5(folder: Path, stem: str, scorer: str | None) -> Path | None:
    candidates = [p for p in folder.glob(f"{stem}DLC*.h5") if not p.stem.endswith(("_filtered", "_skeleton"))]
    if scorer:
        candidates = [p for p in candidates if scorer in p.name]
    if len(candidates) > 1:
        names = ", ".join(p.name for p in candidates)
        raise SystemExit(f"{stem}: several networks analyzed this video ({names}); choose one with --scorer")
    return candidates[0] if candidates else None


def read_prediction(h5: Path) -> tuple[list[str], np.ndarray, np.ndarray]:
    """Bodyparts, (frames, bodyparts, 2) coordinates and (frames, bodyparts) likelihoods."""
    df = pd.read_hdf(h5)
    bodyparts = list(dict.fromkeys(df.columns.get_level_values("bodyparts")))
    df.columns = df.columns.droplevel("scorer")
    xy = np.stack([df[bp][["x", "y"]].values for bp in bodyparts], axis=1)
    likelihood = np.stack([df[bp]["likelihood"].values for bp in bodyparts], axis=1)
    return bodyparts, xy, likelihood


def shape_score(xy: np.ndarray) -> np.ndarray:
    """How far the frame's skeleton is from the animal's usual shape in this video, scaled to 0..1.

    For every pair of bodyparts, the distance between them is compared with its median over the video, in units of
    its own spread (median absolute deviation), so rigid pairs such as the ears weigh more than flexible ones such as
    snout to tail. Time is not used: some videos repeat frames, which makes frame-to-frame jumps useless.
    """
    pairs = [(i, j) for i in range(xy.shape[1]) for j in range(i + 1, xy.shape[1])]
    distance = np.stack([np.hypot(*(xy[:, i] - xy[:, j]).T) for i, j in pairs], axis=1)
    median = np.nanmedian(distance, axis=0)
    spread = np.maximum(np.nanmedian(np.abs(distance - median), axis=0) * 1.4826, 1e-6)
    return np.minimum(np.nan_to_num(np.abs(distance - median) / spread).max(axis=1) / SHAPE_Z, 1.0)


def pick(score: np.ndarray, allowed: np.ndarray, count: int, min_gap: int, taken: set[int]) -> list[int]:
    order = [int(i) for i in np.argsort(-score) if allowed[i]]
    chosen: list[int] = []
    for index in order:
        if all(abs(index - other) >= min_gap for other in [*chosen, *taken]):
            chosen.append(index)
            if len(chosen) == count:
                break
    return sorted(chosen)


def labeled_frames(folder: Path) -> set[int]:
    """Frame numbers of the images in a labeled-data folder (imgNNNN.png, any padding)."""
    return {int(m.group(1)) for p in folder.glob("img*.png") if (m := re.fullmatch(r"img(\d+)", p.stem))} if folder.is_dir() else set()


def write_prelabels(target: Path, frames: list[int], pad: int, prediction: tuple, scorer: str, wanted_parts: list[str], cutoff: float):
    """Adds the predictions of frames to target's CollectedData file, in DLC's labeled-data format.

    Points below cutoff are left empty: a lost point usually sits far from the animal, and placing it is one click.
    """
    bodyparts, xy, lik = prediction
    xy = np.where((lik < cutoff)[..., None], np.nan, xy)
    columns = pd.MultiIndex.from_product([[scorer], wanted_parts, ["x", "y"]], names=["scorer", "bodyparts", "coords"])
    index = pd.MultiIndex.from_arrays([["labeled-data"] * len(frames), [target.name] * len(frames), [f"img{f:0{pad}d}.png" for f in frames]])
    data = np.full((len(frames), len(columns)), np.nan)
    for n, part in enumerate(wanted_parts):
        if part in bodyparts:
            data[:, 2 * n : 2 * n + 2] = xy[frames, bodyparts.index(part)]
    labels = pd.DataFrame(data, index=index, columns=columns)
    label_file = target / f"CollectedData_{scorer}.h5"
    if label_file.exists():
        existing = pd.read_hdf(label_file)
        labels = pd.concat([existing, labels[~labels.index.isin(existing.index)]]).sort_index()
    labels.to_hdf(label_file, key="df_with_missing", mode="w")
    labels.to_csv(label_file.with_suffix(".csv"))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("folders", nargs="+", help="Folders with videos and their DLC .h5 files.")
    parser.add_argument("--out", required=True, help="Folder that receives one subfolder of frames per video (with --apply).")
    parser.add_argument("--project", help="config.yaml of the network the frames are for: skips frames it already has.")
    parser.add_argument("--compare", nargs="+", default=[], help="Folders with a second network's .h5 files of the same videos.")
    parser.add_argument("--scorer", help="Part of the scorer name to use when several networks analyzed a video.")
    parser.add_argument("--count", type=int, default=10, help="Frames per video (default 10).")
    parser.add_argument("--total", type=int, help="Keep only the TOTAL highest scoring frames over all videos.")
    parser.add_argument("--start", type=float, default=0.1, help="Share of each video skipped at the start (default 0.1).")
    parser.add_argument("--end", type=float, default=1.0, help="Share of each video where picking stops (default 1.0).")
    parser.add_argument("--min-gap", type=int, default=30, help="Frames between picks (default 30).")
    parser.add_argument("--absent", type=float, default=0.3, help="Skip frames where the mean likelihood is below this.")
    parser.add_argument("--apply", action="store_true", help="Write the frames and pre-labels; without it, only statistics.")
    args = parser.parse_args()

    project = None
    if args.project:
        with open(args.project, encoding="utf-8") as handle:
            cfg = yaml.safe_load(handle)
        project = {"root": Path(args.project).parent, "scorer": cfg["scorer"], "bodyparts": list(cfg["bodyparts"])}
    cutoff = float(cfg.get("pcutoff", 0.6)) if project else 0.6  # "unsure" in the statistics

    picks = []  # (score, video, frame, unsure, shape, disagree)
    predictions = {}
    for folder in map(Path, args.folders):
        for video in sorted(p for p in folder.iterdir() if p.suffix.lower() in VIDEO_SUFFIXES and not p.stem.endswith("_labeled")):
            stem = video.stem
            h5 = find_h5(folder, stem, args.scorer)
            if h5 is None:
                print(f"skip {video.name}: no .h5 next to it")
                continue
            bodyparts, xy, lik = read_prediction(h5)
            own = (bodyparts, xy, lik)  # pre-labels come from this network even with --compare
            n = len(xy)
            unsure = 1 - lik.min(axis=1)
            shape = shape_score(xy)
            disagree = np.zeros(n)
            compare_h5 = next((h for d in args.compare if (h := find_h5(Path(d), stem, None))), None)
            if args.compare and compare_h5 is None:
                print(f"skip {video.name}: not in --compare folders")
                continue
            if compare_h5:
                _, xy_b, lik_b = read_prediction(compare_h5)
                n = min(n, len(xy_b))
                unsure = np.maximum(unsure[:n], 1 - lik_b[:n].min(axis=1))
                disagree = np.minimum(np.hypot(*(xy[:n] - xy_b[:n]).transpose(2, 0, 1)).max(axis=1) / DISAGREE_PX, 1.0)
                shape, lik = shape[:n], np.maximum(lik[:n], lik_b[:n])
            score = unsure + shape + disagree
            present = lik.mean(axis=1) >= args.absent
            position = np.arange(n) / n
            in_window = (position >= args.start) & (position < args.end)
            taken = labeled_frames(project["root"] / "labeled-data" / stem) if project else set()
            taken |= labeled_frames(Path(args.out) / stem)

            weak = (lik.min(axis=1) < cutoff).astype(float)
            tenths = " ".join(f"{100 * weak[(position >= i / 10) & (position < (i + 1) / 10)].mean():.0f}" for i in range(10))
            chosen = pick(score, in_window & present, args.count, args.min_gap, taken)
            print(
                f"{stem}: {n} frames | unsure % per tenth: {tenths} | absent {100 * (~present).mean():.1f}% | odd shape: {100 * (shape > 0.6).mean():.1f}%"
            )
            if not chosen:
                print("  picked 0 frames")
                continue
            print(
                f"  picked {len(chosen)} (already labeled: {len(taken)}), score {score[chosen].min():.2f}..{score[chosen].max():.2f}, "
                f"by unsure {unsure[chosen].mean():.2f} shape {shape[chosen].mean():.2f}"
                + (f" disagree {disagree[chosen].mean():.2f}" if compare_h5 else "")
            )
            picks += [(float(score[f]), video, f, float(unsure[f]), float(shape[f]), float(disagree[f])) for f in chosen]
            predictions[video] = (own, len(str(n)))

    if args.total and len(picks) > args.total:
        picks = sorted(picks, key=lambda p: -p[0])[: args.total]
    print(f"\ntotal: {len(picks)} frames from {len({p[1] for p in picks})} videos")
    if not args.apply:
        print("statistics only: add --apply to write the frames")
        return

    import cv2

    out_dir = Path(args.out)
    scorer = project["scorer"] if project else "prelabel"
    for video in dict.fromkeys(p[1] for p in picks):
        wanted = sorted({p[2] for p in picks if p[1] == video})
        own, pad = predictions[video]
        target = out_dir / video.stem
        target.mkdir(parents=True, exist_ok=True)
        capture = cv2.VideoCapture(str(video))
        for frame in range(wanted[-1] + 1):  # sequential decode: seeking is not frame-exact
            if not capture.grab():
                raise SystemExit(f"{video.name}: video ended at frame {frame}")
            if frame in wanted:
                cv2.imwrite(str(target / f"img{frame:0{pad}d}.png"), capture.retrieve()[1])
        capture.release()
        write_prelabels(target, wanted, pad, own, scorer, project["bodyparts"] if project else own[0], LOST)
        print(f"wrote {len(wanted)} frames: {target}")

    manifest = out_dir / "hard_manifest.csv"
    new = not manifest.exists()
    with open(manifest, "a", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        if new:
            writer.writerow(["video", "frame", "score", "unsure", "shape", "disagree"])
        writer.writerows(
            [str(p[1]), p[2], f"{p[0]:.3f}", f"{p[3]:.3f}", f"{p[4]:.3f}", f"{p[5]:.3f}"]
            for p in sorted(picks, key=lambda p: (str(p[1]), p[2]))
        )


if __name__ == "__main__":
    main()

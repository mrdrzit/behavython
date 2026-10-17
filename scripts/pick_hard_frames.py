"""
Picks the hardest frames of analyzed videos for a hand-labeled test set.

Two networks analyzed the same videos. A frame is hard when either network is unsure of one of its
bodyparts or when the two networks put a bodypart in different places, so neither network gets an
easier test than the other. Frames come only from the --start..--end part of each video, are spread
at least --min-gap frames apart, and are skipped when neither network sees the animal.

  python scripts\\pick_hard_frames.py <videos_dir> <analysis_dir_A> <analysis_dir_B> <out_dir> [--apply]

Without --apply it only prints the statistics. With it, the frames are written as
<out_dir>/<video stem>/imgNNNN.png (NNNN = frame number) plus <out_dir>/hard_manifest.csv.
Videos are only read, never changed.
"""

import argparse
import csv
from pathlib import Path

import numpy as np
import pandas as pd


def read_prediction(analysis_dir: Path, stem: str) -> tuple[np.ndarray, np.ndarray]:
    """(frames, bodyparts, 2) coordinates and (frames, bodyparts) likelihoods from the unfiltered h5."""
    candidates = [p for p in analysis_dir.glob(f"{stem}DLC*.h5") if "_filtered" not in p.name]
    if len(candidates) != 1:
        raise SystemExit(f"{stem}: expected one unfiltered h5 in {analysis_dir}, found {len(candidates)}")
    df = pd.read_hdf(candidates[0])
    bodyparts = list(dict.fromkeys(df.columns.get_level_values("bodyparts")))
    df.columns = df.columns.droplevel("scorer")
    xy = np.stack([df[bp][["x", "y"]].values for bp in bodyparts], axis=1)
    likelihood = np.stack([df[bp]["likelihood"].values for bp in bodyparts], axis=1)
    return xy, likelihood


def frame_score(xy_a, lik_a, xy_b, lik_b) -> np.ndarray:
    """Higher = harder. Either network unsure, or the two networks disagree."""
    unsure = np.maximum(1 - lik_a.min(axis=1), 1 - lik_b.min(axis=1))
    disagreement = np.hypot(*(xy_a - xy_b).transpose(2, 0, 1)).max(axis=1)
    return unsure + np.minimum(disagreement / 20.0, 1.0)


def pick(score: np.ndarray, allowed: np.ndarray, count: int, min_gap: int) -> list[int]:
    order = [int(i) for i in np.argsort(-score) if allowed[i]]
    chosen: list[int] = []
    for index in order:
        if all(abs(index - other) >= min_gap for other in chosen):
            chosen.append(index)
            if len(chosen) == count:
                break
    return sorted(chosen)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("videos")
    parser.add_argument("analysis_a")
    parser.add_argument("analysis_b")
    parser.add_argument("out_dir")
    parser.add_argument("--count", type=int, default=40)
    parser.add_argument("--start", type=float, default=0.4)
    parser.add_argument("--end", type=float, default=0.8)
    parser.add_argument("--min-gap", type=int, default=30)
    parser.add_argument("--absent", type=float, default=0.3, help="Skip frames where both networks have a mean likelihood below this.")
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()

    videos = sorted(Path(args.videos).glob("*.mp4"))
    manifest = []
    for video in videos:
        stem = video.stem
        try:
            xy_a, lik_a = read_prediction(Path(args.analysis_a), stem)
            xy_b, lik_b = read_prediction(Path(args.analysis_b), stem)
        except SystemExit as error:
            print(f"skip {error}")
            continue
        n = min(len(xy_a), len(xy_b))
        score = frame_score(xy_a[:n], lik_a[:n], xy_b[:n], lik_b[:n])
        present = (lik_a[:n].mean(axis=1) >= args.absent) | (lik_b[:n].mean(axis=1) >= args.absent)
        position = np.arange(n) / n
        in_window = (position >= args.start) & (position < args.end)

        # Where the video is worst: share of frames either network is unsure of, per tenth of the video
        weak = ((lik_a[:n].min(axis=1) < 0.6) | (lik_b[:n].min(axis=1) < 0.6)).astype(float)
        tenths = [f"{100 * weak[(position >= i / 10) & (position < (i + 1) / 10)].mean():.0f}" for i in range(10)]
        print(f"{stem}: {n} frames, unsure % per tenth: {' '.join(tenths)}  | animal absent: {100 * (~present).mean():.1f}%")

        chosen = pick(score, in_window & present, args.count, args.min_gap)
        manifest.extend((stem, frame, float(score[frame]), float(lik_a[frame].min()), float(lik_b[frame].min())) for frame in chosen)
        print(f"  picked {len(chosen)} frames, score {score[chosen].min():.2f}..{score[chosen].max():.2f}")

    if not args.apply:
        return

    import cv2

    out_dir = Path(args.out_dir)
    for stem in dict.fromkeys(row[0] for row in manifest):
        wanted = {row[1] for row in manifest if row[0] == stem}
        target = out_dir / stem
        target.mkdir(parents=True, exist_ok=True)
        capture = cv2.VideoCapture(str(Path(args.videos) / f"{stem}.mp4"))
        for frame in range(max(wanted) + 1):  # sequential decode: seeking is not frame-exact
            ok, image = capture.read()
            if not ok:
                raise SystemExit(f"{stem}: video ended at frame {frame}")
            if frame in wanted:
                cv2.imwrite(str(target / f"img{frame:04d}.png"), image)
        capture.release()
        print(f"wrote {len(wanted)} frames: {target}")
    with open(out_dir / "hard_manifest.csv", "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["stem", "frame", "score", "min_likelihood_a", "min_likelihood_b"])
        writer.writerows(manifest)


if __name__ == "__main__":
    main()

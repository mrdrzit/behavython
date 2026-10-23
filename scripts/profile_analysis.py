"""
Profiles DeepLabCut tracking results (filtered .csv or .h5, or unfiltered .h5) of one or many analyzed videos.

Per video: how often each bodypart is below the likelihood cutoff, how often it jumps further than a
threshold between consecutive frames, the share of frames where any bodypart is unsure, the longest run of
such frames, and where in the video (per tenth) they cluster. Videos are grouped by folder and scorer, so
pointing it at the analysis folders of two networks gives one summary per network to compare.

  python scripts\\profile_analysis.py <folder or .h5 file> [more ...] [--pcutoff 0.6] [--jump 20] [--out DIR]

Folders are searched recursively. --source auto (default) uses, per video, the _filtered .csv, else the
_filtered .h5, else the unfiltered .h5; filtered or unfiltered force one kind. Skeleton and
image-prediction files are skipped. Videos and tracking files are only read. Filtered files are smoothed,
so jump and unsure numbers are not comparable with unfiltered ones. With --out it writes analysis_profile.csv (one row per video and bodypart) and one
analysis_profile_<n>.png per folder and scorer (unsure % per video, unsure % along each video, unsure and jump %
per bodypart); --no-plots skips the PNGs.
"""

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

SKIPPED = ("_skeleton", "image_predictions", "_full", "_meta")


def base_name(path: Path) -> str:
    return path.stem.removesuffix("_filtered")


def find_h5(paths: list[str], source: str = "auto") -> list[Path]:
    """source: unfiltered (.h5 only), filtered (_filtered .h5 or .csv), auto (filtered csv, filtered h5, unfiltered)."""
    unfiltered, filtered = [], []
    for text in paths:
        path = Path(text)
        candidates = [path] if path.is_file() else sorted([*path.rglob("*.h5"), *path.rglob("*.csv")])
        for p in candidates:
            if any(tag in p.name for tag in SKIPPED):
                continue
            if p.stem.endswith("_filtered"):
                filtered.append(p)
            elif p.suffix == ".h5":
                unfiltered.append(p)
    unfiltered, filtered = list(dict.fromkeys(unfiltered)), list(dict.fromkeys(filtered))
    if source == "unfiltered":
        return unfiltered
    if source == "filtered":
        return filtered
    # Per video: filtered .csv first, then filtered .h5, then unfiltered .h5.
    chosen = {}
    for p in sorted(filtered, key=lambda p: p.suffix != ".csv") + unfiltered:
        chosen.setdefault((p.parent, base_name(p)), p)
    return list(chosen.values())


def read_tracking(path: Path) -> pd.DataFrame:
    if path.suffix == ".csv":
        df = pd.read_csv(path, header=[0, 1, 2], index_col=0)
        df.columns.names = ["scorer", "bodyparts", "coords"]
        return df
    return pd.read_hdf(path)


def longest_run(flags: np.ndarray) -> int:
    best = current = 0
    for flag in flags:
        current = current + 1 if flag else 0
        best = max(best, current)
    return best


def profile_file(path: Path, pcutoff: float, jump: float) -> dict | None:
    df = read_tracking(path)
    if "likelihood" not in df.columns.get_level_values("coords"):
        return None
    scorer = df.columns.get_level_values("scorer")[0]
    df.columns = df.columns.droplevel("scorer")
    bodyparts = list(dict.fromkeys(df.columns.get_level_values("bodyparts")))
    frames = len(df)
    likelihood = np.stack([df[bp]["likelihood"].to_numpy() for bp in bodyparts], axis=1)
    xy = np.stack([df[bp][["x", "y"]].to_numpy() for bp in bodyparts], axis=1)
    low = likelihood < pcutoff
    step = np.hypot(*np.diff(xy, axis=0).transpose(2, 0, 1))
    jumped = step > jump
    any_low = low.any(axis=1)
    position = np.arange(frames) / frames
    tenths = [100 * any_low[(position >= i / 10) & (position < (i + 1) / 10)].mean() for i in range(10)]
    stem = path.name.split("DLC")[0] if "DLC" in path.name else base_name(path)
    return {
        "video": stem,
        "scorer": scorer,
        "folder": str(path.parent),
        "frames": frames,
        "bodyparts": bodyparts,
        "low_pct": {bp: 100 * low[:, i].mean() for i, bp in enumerate(bodyparts)},
        "median_likelihood": {bp: float(np.median(likelihood[:, i])) for i, bp in enumerate(bodyparts)},
        "jump_pct": {bp: 100 * jumped[:, i].mean() for i, bp in enumerate(bodyparts)},
        "any_low_pct": 100 * any_low.mean(),
        "longest_low_run": longest_run(any_low),
        "tenths": tenths,
    }


def plot_group(items: list[dict], title: str, path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    bodyparts = items[0]["bodyparts"]
    videos = [i["video"] for i in items]
    height = max(3.5, 0.35 * len(items) + 1.5)
    fig, axes = plt.subplots(1, 3, figsize=(16, height), constrained_layout=True)
    fig.suptitle(title, fontsize=10)

    axes[0].barh(videos, [i["any_low_pct"] for i in items], color="tab:red")
    axes[0].invert_yaxis()
    axes[0].set_xlabel("frames with any bodypart unsure (%)")

    image = axes[1].imshow([i["tenths"] for i in items], aspect="auto", cmap="Reds", vmin=0, vmax=100)
    axes[1].set_yticks(range(len(items)), videos)
    axes[1].set_xticks(range(10), [f"{i * 10}" for i in range(10)])
    axes[1].set_xlabel("position in video (% start of tenth)")
    fig.colorbar(image, ax=axes[1], label="unsure (%)")

    x = np.arange(len(bodyparts))
    weights = np.array([i["frames"] for i in items], dtype=float)
    low = [np.average([i["low_pct"][bp] for i in items], weights=weights) for bp in bodyparts]
    jumps = [np.average([i["jump_pct"][bp] for i in items], weights=weights) for bp in bodyparts]
    axes[2].bar(x - 0.2, low, 0.4, label="unsure", color="tab:red")
    axes[2].bar(x + 0.2, jumps, 0.4, label="jump", color="tab:blue")
    axes[2].set_xticks(x, bodyparts, rotation=30, ha="right")
    axes[2].set_ylabel("% of frames (all videos)")
    axes[2].legend()

    fig.savefig(path, dpi=130)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("paths", nargs="+")
    parser.add_argument("--pcutoff", type=float, default=0.6, help="Likelihood below this counts as unsure.")
    parser.add_argument("--jump", type=float, default=20.0, help="Frame-to-frame movement in px that counts as a jump.")
    parser.add_argument("--out", help="Folder for analysis_profile.csv.")
    parser.add_argument(
        "--source",
        choices=["auto", "unfiltered", "filtered"],
        default="auto",
        help="auto: per video, the _filtered .csv, else the _filtered .h5, else the unfiltered .h5. "
        "filtered: _filtered .csv/.h5 only. unfiltered: unfiltered .h5 only.",
    )
    parser.add_argument("--no-plots", action="store_true", help="With --out, write only the csv.")
    args = parser.parse_args()

    files = find_h5(args.paths, args.source)
    if not files:
        raise SystemExit(f"No tracking files found (source={args.source}).")
    profiles = [p for p in (profile_file(f, args.pcutoff, args.jump) for f in files) if p]
    print(f"{len(profiles)} analyses | unsure = likelihood < {args.pcutoff} | jump = more than {args.jump} px between frames\n")

    groups = defaultdict(list)
    for item in profiles:
        groups[(item["folder"], item["scorer"])].append(item)

    for (folder, scorer), items in groups.items():
        bodyparts = items[0]["bodyparts"]
        print(f"=== {scorer}\n    {folder}")
        header = (
            f"{'video':<34}{'frames':>7}{'any unsure %':>13}{'longest run':>12}  "
            + "  ".join(f"{bp[:7]:>7}" for bp in bodyparts)
            + "   worst tenths of video (unsure %)"
        )
        print(header)
        for item in items:
            cells = "  ".join(f"{item['low_pct'][bp]:>7.1f}" for bp in bodyparts)
            worst = sorted(range(10), key=lambda i: -item["tenths"][i])[:2]
            where = ", ".join(f"{i * 10}-{i * 10 + 10}%: {item['tenths'][i]:.0f}" for i in sorted(worst))
            print(f"{item['video']:<34}{item['frames']:>7}{item['any_low_pct']:>13.1f}{item['longest_low_run']:>12}  {cells}   {where}")
        total = sum(i["frames"] for i in items)
        any_low = sum(i["any_low_pct"] * i["frames"] for i in items) / total
        cells = "  ".join(f"{sum(i['low_pct'][bp] * i['frames'] for i in items) / total:>7.1f}" for bp in bodyparts)
        print(f"{'ALL (frame-weighted)':<34}{total:>7}{any_low:>13.1f}{max(i['longest_low_run'] for i in items):>12}  {cells}")
        jumps = "  ".join(f"{bp[:7]} {sum(i['jump_pct'][bp] * i['frames'] for i in items) / total:.2f}" for bp in bodyparts)
        median = "  ".join(f"{bp[:7]} {np.median([i['median_likelihood'][bp] for i in items]):.2f}" for bp in bodyparts)
        print(f"    jumps over {args.jump:.0f} px, % of frames: {jumps}")
        print(f"    median likelihood (median over videos):  {median}\n")

    if args.out:
        out = Path(args.out)
        out.mkdir(parents=True, exist_ok=True)
        with open(out / "analysis_profile.csv", "w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(
                [
                    "scorer",
                    "folder",
                    "video",
                    "frames",
                    "bodypart",
                    f"pct_below_{args.pcutoff}",
                    "median_likelihood",
                    f"pct_jump_over_{args.jump:g}px",
                    "pct_frames_any_unsure",
                    "longest_unsure_run",
                ]
            )
            for item in profiles:
                for bp in item["bodyparts"]:
                    writer.writerow(
                        [
                            item["scorer"],
                            item["folder"],
                            item["video"],
                            item["frames"],
                            bp,
                            f"{item['low_pct'][bp]:.3f}",
                            f"{item['median_likelihood'][bp]:.4f}",
                            f"{item['jump_pct'][bp]:.3f}",
                            f"{item['any_low_pct']:.3f}",
                            item["longest_low_run"],
                        ]
                    )
        print(f"wrote {out / 'analysis_profile.csv'}")
        if not args.no_plots:
            for n, ((folder, scorer), items) in enumerate(groups.items(), 1):
                png = out / f"analysis_profile_{n}.png"
                plot_group(items, f"{scorer} | {folder} | unsure < {args.pcutoff}, jump > {args.jump:g} px", png)
                print(f"wrote {png}")


if __name__ == "__main__":
    main()

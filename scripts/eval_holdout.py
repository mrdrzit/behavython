"""
Compares DeepLabCut networks and snapshots on hand-labeled holdout frames.

The truth is every folder logged in <project>/review_log.csv with one of the --rounds tags
(default: holdout, hard_holdout), read from <project>/labeled-data. Each model is run on those
frames with deeplabcut's analyze_images and scored against the labels, separately per round and
per phase (pre/pos-pintura, taken from manifest.csv, or "pos" when a stem is not listed).

Models are given as LABEL=CONFIG[:SNAPSHOT,SNAPSHOT,...], with snapshot names such as snapshot-300
(default: the last one). Predictions are cached in <project>/analysis/holdout_eval, so adding a
snapshot only runs the new one.

  python scripts\\eval_holdout.py <project> ^
      --model sep18=E:\\...\\CD1-GUSTAVO-2026-09-18\\config.yaml:snapshot-500 ^
      --model oct12=<project>\\config.yaml:snapshot-best-160,snapshot-200,snapshot-300,snapshot-350
"""

import argparse
import csv
import importlib
import pickle
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg"}
THRESHOLDS_PX = (5, 10)


def load_truth(project: Path, rounds: list[str]) -> pd.DataFrame:
    """Long table of labeled points: round, phase, folder, image, bodypart, x, y."""
    phase_of = {}
    manifest = project / "manifest.csv"
    if manifest.exists():
        with open(manifest, encoding="utf-8") as handle:
            phase_of = {row["stem"]: "pre" if row["group"].startswith("pre") else "pos" for row in csv.DictReader(handle)}

    rows = []
    with open(project / "review_log.csv", encoding="utf-8") as handle:
        logged = [(row["round"], row["folder"]) for row in csv.DictReader(handle) if row["round"] in rounds]
    for round_name, folder in dict.fromkeys(logged):
        label_file = next((project / "labeled-data" / folder).glob("CollectedData_*.h5"))
        df = pd.read_hdf(label_file)
        bodyparts = list(dict.fromkeys(df.columns.get_level_values("bodyparts")))
        for index, values in df.iterrows():
            image = project / "labeled-data" / folder / Path(str(index[-1] if isinstance(index, tuple) else index)).name
            for bodypart in bodyparts:
                x = values.xs((bodypart, "x"), level=("bodyparts", "coords")).iloc[0]
                y = values.xs((bodypart, "y"), level=("bodyparts", "coords")).iloc[0]
                if pd.notna(x) and pd.notna(y):
                    rows.append((round_name, phase_of.get(folder, "pos"), folder, str(image), bodypart, float(x), float(y)))
    return pd.DataFrame(rows, columns=["round", "phase", "folder", "image", "bodypart", "x", "y"])


def run_model(config: Path, shuffle: int, snapshot: str | None, images: list[str], cache: Path) -> dict[str, np.ndarray]:
    """Returns {image path: (n_bodyparts, 3) array of x, y, likelihood} for one snapshot."""
    result = {}
    if cache.exists():
        with open(cache, "rb") as handle:
            result = pickle.load(handle)
        images = [image for image in images if str(Path(image)) not in result]
        if not images:
            return result

    from deeplabcut.pose_estimation_pytorch.apis.utils import get_model_snapshots
    from deeplabcut.pose_estimation_pytorch.task import Task
    from deeplabcut.utils import auxiliaryfunctions
    from deeplabcut.core.engine import Engine

    cfg = auxiliaryfunctions.read_config(str(config))
    model_folder = Path(cfg["project_path"]) / auxiliaryfunctions.get_model_folder(
        cfg["TrainingFraction"][0], shuffle, cfg, engine=Engine.PYTORCH
    )
    train_folder = model_folder / "train"
    snapshots = get_model_snapshots("all", train_folder, Task.BOTTOM_UP)
    names = [s.path.stem for s in snapshots]
    index = len(names) - 1 if snapshot is None else names.index(snapshot)
    print(f"  {config.parent.name}: {names[index]} (index {index} of {names})")

    analyze = importlib.import_module("deeplabcut.pose_estimation_pytorch.apis.analyze_images").analyze_images
    # analyze_images writes a predictions file next to the images unless told otherwise; keep it out of labeled-data
    predictions = analyze(
        str(config), images, shuffle=shuffle, snapshot_index=index, output_dir=cache.parent / "raw", progress_bar=False, plotting=False
    )
    for image, prediction in predictions.items():
        result[str(Path(image))] = np.asarray(prediction["bodyparts"])[0]
    result["__bodyparts__"] = list(cfg["bodyparts"])
    cache.parent.mkdir(parents=True, exist_ok=True)
    with open(cache, "wb") as handle:
        pickle.dump(result, handle)
    return result


def score(truth: pd.DataFrame, predictions: dict, pcutoff: float) -> pd.DataFrame:
    bodyparts = predictions["__bodyparts__"]
    errors = []
    likelihoods = []
    for row in truth.itertuples():
        prediction = predictions.get(str(Path(row.image)))
        if prediction is None or row.bodypart not in bodyparts:
            errors.append(np.nan)
            likelihoods.append(np.nan)
            continue
        px, py, likelihood = prediction[bodyparts.index(row.bodypart)]
        errors.append(float(np.hypot(px - row.x, py - row.y)))
        likelihoods.append(float(likelihood))
    scored = truth.copy()
    scored["error"] = errors
    scored["likelihood"] = likelihoods
    return scored


def summarize(scored: pd.DataFrame, pcutoff: float) -> dict:
    error = scored["error"].dropna()
    summary = {"points": len(error), "rmse": float(np.sqrt((error**2).mean())), "median": float(error.median())}
    for threshold in THRESHOLDS_PX:
        summary[f"within{threshold}px_%"] = float((error <= threshold).mean() * 100)
    summary[f"lowconf<{pcutoff}_%"] = float((scored["likelihood"].dropna() < pcutoff).mean() * 100)
    return summary


def parse_model(text: str) -> tuple[str, Path, list[str | None]]:
    label, _, rest = text.partition("=")
    # config path may contain a drive colon, so snapshots are split off the last colon after .yaml
    marker = rest.lower().rfind(".yaml:")
    if marker == -1:
        return label, Path(rest), [None]
    return label, Path(rest[: marker + 5]), rest[marker + 6 :].split(",")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("project", help="Project holding the labeled holdout (the new network).")
    parser.add_argument("--model", action="append", required=True, help="LABEL=CONFIG[:SNAPSHOT,...]")
    parser.add_argument("--rounds", nargs="+", default=["holdout", "hard_holdout"])
    parser.add_argument("--shuffle", type=int, default=1)
    parser.add_argument("--pcutoff", type=float, default=0.6)
    args = parser.parse_args()
    warnings.filterwarnings("ignore")

    project = Path(args.project)
    truth = load_truth(project, args.rounds)
    images = sorted(truth["image"].unique())
    print(f"truth: {len(truth)} points, {len(images)} images, rounds {sorted(truth['round'].unique())}")

    out_dir = project / "analysis" / "holdout_eval"
    summaries, per_bodypart = [], []
    for text in args.model:
        label, config, snapshots = parse_model(text)
        for snapshot in snapshots:
            cache = out_dir / f"{label}_{snapshot or 'last'}.pkl"
            predictions = run_model(config, args.shuffle, snapshot, images, cache)
            scored = score(truth, predictions, args.pcutoff)
            name = f"{label}:{snapshot or 'last'}"
            for (round_name, phase), group in scored.groupby(["round", "phase"]):
                summaries.append({"model": name, "round": round_name, "phase": phase, **summarize(group, args.pcutoff)})
            for (round_name, phase, bodypart), group in scored.groupby(["round", "phase", "bodypart"]):
                per_bodypart.append(
                    {
                        "model": name,
                        "round": round_name,
                        "phase": phase,
                        "bodypart": bodypart,
                        "median": group["error"].median(),
                        "rmse": float(np.sqrt((group["error"] ** 2).mean())),
                    }
                )

    summary = pd.DataFrame(summaries)
    detail = pd.DataFrame(per_bodypart)
    summary.to_csv(out_dir / "summary.csv", index=False)
    detail.to_csv(out_dir / "per_bodypart.csv", index=False)
    pd.set_option("display.width", 200, "display.float_format", "{:.2f}".format)
    for (round_name, phase), group in summary.groupby(["round", "phase"]):
        print(f"\n=== {round_name} / {phase}")
        print(group.drop(columns=["round", "phase"]).to_string(index=False))
    print("\nPer-bodypart RMSE (px):")
    print(detail.pivot_table(index=["round", "phase", "bodypart"], columns="model", values="rmse").to_string())


if __name__ == "__main__":
    main()

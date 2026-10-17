"""
Profiles a trained DeepLabCut (PyTorch) network from what training left on disk: train.txt,
learning_stats.csv, pytorch_config.yaml and the snapshot files.

Reports the configuration that ran, how long it took, the test metrics every evaluation produced, the best
epochs, and which snapshots exist, which were expected from save_epochs and which were rotated away by
max_snapshots. Nothing in the network folder is modified.

  python scripts\\profile_training.py <train folder | project folder> [--out DIR]

A project folder profiles every dlc-models-pytorch\\iteration-*\\*\\train it contains. With --out it also
writes training_profile.txt, training_epochs.csv and training_profile.png (one set per network, in a
subfolder named after the model).
"""

import argparse
import csv
import io
import re
import statistics
from contextlib import redirect_stdout
from datetime import datetime
from itertools import pairwise
from pathlib import Path

import yaml

EPOCH_LINE = re.compile(
    r"^(\d{4}-\d\d-\d\d \d\d:\d\d:\d\d) Epoch (\d+)/(\d+) \(lr=([\d.e+-]+)\), train loss ([\d.]+)"
    r"(?:, valid loss ([\d.]+))?, GPU: ([\d.]+)/([\d.]+) MiB"
)
SNAPSHOT = re.compile(r"^snapshot-(?:(best)-)?(\d+)\.pt$")
METRICS = ("test.mAP", "test.mAR", "test.rmse", "test.rmse_pcutoff")


def timestamp(text: str) -> datetime:
    return datetime.strptime(text, "%Y-%m-%d %H:%M:%S").astimezone()


def parse_log(path: Path) -> dict:
    """Epoch lines and session count from train.txt. Only the last training session is reported."""
    sessions, current = 0, []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if "Training with configuration:" in line:
            sessions += 1
            current = []
        match = EPOCH_LINE.match(line)
        if match:
            when, epoch, total, lr, train, valid, gpu_used, gpu_total = match.groups()
            current.append(
                (timestamp(when), int(epoch), int(total), float(lr), float(train), valid and float(valid), float(gpu_used), float(gpu_total))
            )
    return {"sessions": sessions, "epochs": current}


def read_stats(path: Path) -> list[dict]:
    """learning_stats.csv rows, one per epoch (last row wins if an epoch appears twice)."""
    rows = {}
    with open(path, newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            rows[int(float(row["step"]))] = {key: (float(value) if value not in ("", None) else None) for key, value in row.items()}
    return [rows[step] for step in sorted(rows)]


def modified(path: Path) -> str:
    return datetime.fromtimestamp(path.stat().st_mtime).astimezone().strftime("%m-%d %H:%M")


def duration(seconds: float) -> str:
    hours, rest = divmod(int(seconds), 3600)
    return f"{hours} h {rest // 60:02d} min"


def describe_augmentation(data: dict) -> list[str]:
    train = data.get("train", {})
    parts = []
    for key, value in train.items():
        if value in (None, False) or key in ("normalize_images", "scale_to_unit_range", "collate", "top_down_crop"):
            continue
        parts.append(f"{key}={value}" if value is not True else key)
    return parts


def profile(train_dir: Path, out_dir: Path | None) -> None:
    config = yaml.safe_load((train_dir / "pytorch_config.yaml").read_text(encoding="utf-8"))
    log = parse_log(train_dir / "train.txt") if (train_dir / "train.txt").exists() else {"sessions": 0, "epochs": []}
    stats = read_stats(train_dir / "learning_stats.csv") if (train_dir / "learning_stats.csv").exists() else []
    model_name = train_dir.parent.name
    runner, settings = config["runner"], config["train_settings"]
    snapshot_cfg = runner["snapshots"]

    print(f"=== {model_name}  ({train_dir})")
    print(
        f"network       {config.get('net_type')} | method {config.get('method')} | batch {settings['batch_size']} | bodyparts {config['metadata']['bodyparts']}"
    )
    print(f"optimizer     {runner['optimizer']['type']} {runner['optimizer']['params']}")
    scheduler = runner.get("scheduler") or {}
    print(f"scheduler     {scheduler.get('type')} {scheduler.get('params')}")
    print(f"snapshots     save every {snapshot_cfg['save_epochs']} epochs, keep {snapshot_cfg['max_snapshots']}")
    print(f"augmentation  {', '.join(describe_augmentation(config['data'])) or 'none'}")
    print(f"inference     auto_padding {config['data'].get('inference', {}).get('auto_padding')}")

    epochs = log["epochs"]
    first_epoch = epochs[0][1] if epochs else None
    last_epoch = epochs[-1][1] if epochs else (stats[-1]["step"] if stats else None)
    if epochs:
        seconds = (epochs[-1][0] - epochs[0][0]).total_seconds()
        gaps = [(b[0] - a[0]).total_seconds() for a, b in pairwise(epochs)]
        print(f"\nlog           {log['sessions']} training session(s) in train.txt, last one reported")
        print(
            f"run           epochs {first_epoch} to {last_epoch} of {epochs[-1][2]} | {epochs[0][0]:%Y-%m-%d %H:%M} to {epochs[-1][0]:%H:%M} | {duration(seconds)}"
        )
        if gaps:
            print(
                f"epoch time    median {statistics.median(gaps) / 60:.2f} min, mean {statistics.mean(gaps) / 60:.2f} min, max {max(gaps) / 60:.2f} min"
            )
        print(f"GPU memory    peak {max(e[6] for e in epochs) / 1024:.1f} of {epochs[0][7] / 1024:.1f} GB")
        print(f"final loss    train {epochs[-1][4]:.5f}" + (f", valid {epochs[-1][5]:.5f}" if epochs[-1][5] else ""))
    elif last_epoch:
        print(f"\nlog           no epoch lines found in train.txt; last epoch in learning_stats.csv is {int(last_epoch)}")

    evaluated = [row for row in stats if row.get("metrics/test.mAP") is not None]
    if evaluated:
        print(f"\nevaluations   {len(evaluated)} (epochs {int(evaluated[0]['step'])} to {int(evaluated[-1]['step'])})")
        best_map = max(evaluated, key=lambda row: row["metrics/test.mAP"])
        best_rmse = min(evaluated, key=lambda row: row["metrics/test.rmse"])
        last = evaluated[-1]
        for label, row in (("best mAP", best_map), ("best RMSE", best_rmse), ("last", last)):
            print(
                f"  {label:<10} epoch {int(row['step']):>4}  mAP {row['metrics/test.mAP']:.2f}  mAR {row['metrics/test.mAR']:.2f}  RMSE {row['metrics/test.rmse']:.2f}  RMSE(pcutoff) {row['metrics/test.rmse_pcutoff']:.2f}"
            )
        spread = [row["metrics/test.mAP"] for row in evaluated[-5:]]
        print(f"  last 5 evaluations mAP {min(spread):.2f} to {max(spread):.2f}")

    print("\nsnapshots")
    found = {}
    for path in sorted(train_dir.glob("*.pt")):
        match = SNAPSHOT.match(path.name)
        found[path.name] = (match, path)
    eval_by_epoch = {int(row["step"]): row for row in evaluated}
    start = first_epoch if first_epoch is not None else 1
    expected = {e for e in range(start, (last_epoch or 0) + 1) if e % snapshot_cfg["save_epochs"] == 0}
    if last_epoch:
        expected.add(int(last_epoch))
    present = set()
    for name, (match, path) in found.items():
        if not match:
            print(f"  {name:<24} {path.stat().st_size / 1e6:>5.0f} MB  {modified(path)}  (not a training snapshot)")
            continue
        epoch = int(match.group(2))
        if not match.group(1):
            present.add(epoch)
        nearest = min(eval_by_epoch, key=lambda e: abs(e - epoch)) if eval_by_epoch else None
        note = ""
        if nearest is not None and abs(nearest - epoch) <= 9:
            row = eval_by_epoch[nearest]
            note = f"mAP {row['metrics/test.mAP']:.2f} RMSE {row['metrics/test.rmse']:.2f}" + (
                "" if nearest == epoch else f" (evaluated at epoch {nearest})"
            )
        kind = "best" if match.group(1) else "epoch"
        print(f"  {name:<24} {kind:<5} {path.stat().st_size / 1e6:>5.0f} MB  {modified(path)}  {note}")
    rotated = sorted(expected - present)
    print(f"  expected from save_epochs: {sorted(expected)}")
    print(f"  missing (rotated away by max_snapshots or never reached): {rotated or 'none'}")

    if out_dir:
        target = out_dir / model_name
        target.mkdir(parents=True, exist_ok=True)
        with open(target / "training_epochs.csv", "w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(["epoch", "lr", "train_loss", "valid_loss", *METRICS])
            lr_by_epoch = {e[1]: e[3] for e in epochs}
            for row in stats:
                writer.writerow(
                    [
                        int(row["step"]),
                        lr_by_epoch.get(int(row["step"])),
                        row.get("losses/train.total_loss"),
                        row.get("losses/eval.total_loss"),
                        *(row.get(f"metrics/{m}") for m in METRICS),
                    ]
                )
        plot(stats, evaluated, present, target / "training_profile.png", model_name)
        print(f"\nwrote {target}")


def plot(stats: list[dict], evaluated: list[dict], present: set[int], path: Path, title: str) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, (top, bottom) = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
    steps = [row["step"] for row in stats]
    top.plot(steps, [row.get("losses/train.total_loss") for row in stats], label="train loss")
    top.plot(
        [row["step"] for row in evaluated], [row.get("losses/eval.total_loss") for row in evaluated], "o-", label="eval loss", markersize=3
    )
    top.set_yscale("log")
    top.set_ylabel("loss")
    top.legend()
    top.set_title(title)
    bottom.plot([row["step"] for row in evaluated], [row["metrics/test.mAP"] for row in evaluated], "o-", markersize=3, label="test mAP")
    bottom.set_ylabel("test mAP")
    rmse = bottom.twinx()
    rmse.plot(
        [row["step"] for row in evaluated],
        [row["metrics/test.rmse"] for row in evaluated],
        "s-",
        color="tab:red",
        markersize=3,
        label="test RMSE (px)",
    )
    rmse.set_ylabel("test RMSE (px)")
    for epoch in sorted(present):
        bottom.axvline(epoch, color="grey", alpha=0.25, linewidth=0.8)
    bottom.set_xlabel("epoch (grey lines: saved snapshots)")
    figure.tight_layout()
    figure.savefig(path, dpi=130)
    plt.close(figure)


def find_train_dirs(path: Path) -> list[Path]:
    if (path / "pytorch_config.yaml").exists():
        return [path]
    return sorted(p.parent for p in (path / "dlc-models-pytorch").glob("iteration-*/*/train/pytorch_config.yaml"))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("path", help="A train folder, or a project folder.")
    parser.add_argument("--out", help="Folder for training_profile.txt, training_epochs.csv and training_profile.png.")
    args = parser.parse_args()

    train_dirs = find_train_dirs(Path(args.path))
    if not train_dirs:
        raise SystemExit("No train folder with a pytorch_config.yaml found.")
    out_dir = Path(args.out) if args.out else None
    report = io.StringIO()
    for train_dir in train_dirs:
        buffer = io.StringIO()
        with redirect_stdout(buffer):
            profile(train_dir, out_dir)
        print(buffer.getvalue())
        report.write(buffer.getvalue() + "\n")
    if out_dir:
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "training_profile.txt").write_text(report.getvalue(), encoding="utf-8")


if __name__ == "__main__":
    main()

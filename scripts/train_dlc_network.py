"""
Trains a DeepLabCut (PyTorch) network from a terminal and shows its progress live.

Training runs exactly what train/pytorch_config.yaml says (epochs, schedule, snapshots, augmentation); nothing is
overridden, unlike the DLC GUI. It runs in a child process whose console output goes to train/train_console.log,
while this terminal prints one line per finished epoch, read from DLC's own train/train.txt:

  epoch 190/300 | loss 0.00017 | valid 0.00110 | mAP 100.00 | rmse 0.73 (best 0.73 @ 190) | 10 s/epoch | left 18m20s | ETA 22:31

"best" follows the network's runner.key_metric, the same metric DLC uses for snapshot-best (test.rmse for networks
made by make_dlc_network.py, test.mAP otherwise).

Shuffle 1 and the first training fraction, as behavython uses. Ctrl+C stops the watcher and the training.
Closing the terminal also stops the training.

Requirements: a Python env with DeepLabCut 3.0.2 (PyTorch) and PyYAML; behavython itself is not imported. Single-animal
PyTorch projects only (not TensorFlow; multi-animal and top-down not tested). The watcher reads DLC 3.0.2's train.txt
lines ("Epoch N/M (lr=...), train loss X"); another DLC version may log differently and show no epochs.

  scripts\\train_dlc_network.bat <project>                      launcher: uses the behavython miniforge env
  python scripts\\train_dlc_network.py <project folder or config.yaml>
  python scripts\\train_dlc_network.py <project> --watch       only watch a training that is already running
  python scripts\\train_dlc_network.py <project> --overwrite   train although the folder already has snapshots
"""

import argparse
import re
import subprocess
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

import yaml

EPOCH_LINE = re.compile(r"^(\S+ \S+) Epoch (\d+)/(\d+) \(lr=[^)]*\), train loss ([\d.e+-]+)(?:, valid loss ([\d.e+-]+))?")
METRIC_LINE = re.compile(r"metrics/test\.(mAP|rmse):\s+([\d.]+)")
RUN_START = "Training with configuration:"
TIME_WINDOW = 10  # epochs averaged for the time estimate; covers one evaluation at eval_interval 10


def find_train_dir(config_path: Path) -> Path:
    with open(config_path, encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle)
    fraction = round(100 * cfg["TrainingFraction"][0])
    models = config_path.parent / "dlc-models-pytorch" / f"iteration-{cfg['iteration']}"
    matches = sorted(models.glob(f"*trainset{fraction}shuffle1"))
    if len(matches) != 1:
        raise SystemExit(f"expected one *trainset{fraction}shuffle1 folder in {models}, found {len(matches)}")
    return matches[0] / "train"


def read_run(train_txt: Path, offset: int = 0) -> list[dict]:
    """Epochs of the latest run in train.txt after byte offset (DLC appends a new run to the same file)."""
    if not train_txt.exists():
        return []
    with open(train_txt, "rb") as handle:
        handle.seek(offset)
        lines = handle.read().decode("utf-8", errors="replace").splitlines()
    starts = [n for n, line in enumerate(lines) if RUN_START in line]
    epochs = []
    for line in lines[starts[-1] if starts else 0 :]:
        if match := EPOCH_LINE.match(line):
            stamp, epoch, total, loss, valid = match.groups()
            epochs.append(
                {
                    "time": datetime.strptime(stamp, "%Y-%m-%d %H:%M:%S"),  # noqa: DTZ007  (local log time)
                    "epoch": int(epoch),
                    "total": int(total),
                    "loss": float(loss),
                    "valid": float(valid) if valid else None,
                }
            )
        elif (match := METRIC_LINE.search(line)) and epochs:
            # DLC prints the metrics of an evaluation right after that epoch's line
            epochs[-1][match.group(1)] = float(match.group(2))
    return epochs


def format_duration(seconds: float) -> str:
    minutes = int(seconds // 60)
    return f"{minutes // 60}h{minutes % 60:02d}m" if minutes >= 60 else f"{minutes}m{int(seconds % 60):02d}s"


def best_metric(train_dir: Path) -> tuple[str, bool]:
    """The metric DLC uses for snapshot-best (runner.key_metric), and whether larger is better."""
    with open(train_dir / "pytorch_config.yaml", encoding="utf-8") as handle:
        runner = yaml.safe_load(handle).get("runner", {})
    key = str(runner.get("key_metric", "test.mAP")).removeprefix("test.")
    if key not in ("mAP", "rmse"):  # only these two are read from train.txt
        return "mAP", True
    return key, bool(runner.get("key_metric_asc", True))


def format_epoch(epochs: list[dict], index: int, key: str = "mAP", larger_is_better: bool = True) -> str:
    row = epochs[index]
    parts = [f"epoch {row['epoch']:>4}/{row['total']}", f"loss {row['loss']:.5f}"]
    if row["valid"] is not None:
        parts.append(f"valid {row['valid']:.5f}")
    evaluated = [r for r in epochs[: index + 1] if "mAP" in r]
    if evaluated:
        last = evaluated[-1]
        scored = [r for r in evaluated if key in r]
        best = (max if larger_is_better else min)(scored, key=lambda r: r[key]) if scored else None
        for name in ("mAP", "rmse"):
            if name in last:
                text = f"{name} {last[name]:.2f}"
                if name == key and best is not None:
                    text += f" (best {best[key]:.2f} @ {best['epoch']})"
                parts.append(text)
    window = epochs[max(0, index - TIME_WINDOW) : index + 1]
    if len(window) > 1:
        per_epoch = (window[-1]["time"] - window[0]["time"]).total_seconds() / (len(window) - 1)
        left = per_epoch * (row["total"] - row["epoch"])
        parts += [f"{per_epoch:.0f} s/epoch", f"left {format_duration(left)}"]
        parts.append(f"ETA {(row['time'] + timedelta(seconds=left)):%H:%M}")
    return " | ".join(parts)


def watch(train_dir: Path, process: subprocess.Popen | None, poll: float, offset: int = 0) -> None:
    train_txt = train_dir / "train.txt"
    key, larger_is_better = best_metric(train_dir)
    printed = 0
    started = time.time()
    said_waiting = False
    while True:
        epochs = read_run(train_txt, offset)
        if printed > len(epochs):  # a new run started in the same file
            printed = 0
        ready = len(epochs)
        ended = process is not None and process.poll() is not None
        if epochs and epochs[-1]["valid"] is not None and "mAP" not in epochs[-1] and not ended:
            ready -= 1  # evaluation epoch whose metrics are not written yet: print it on the next read
        for index in range(printed, ready):
            print(format_epoch(epochs, index, key, larger_is_better), flush=True)
        printed = max(printed, ready)
        if printed and epochs[printed - 1]["epoch"] == epochs[printed - 1]["total"]:
            print("training finished")
            return
        if ended:
            print(f"training process ended with exit code {process.returncode}; see {train_dir / 'train_console.log'}")
            return
        if not epochs and not said_waiting and time.time() - started > 30:
            said_waiting = True
            print("waiting for the first epoch (loading the dataset and the first epoch take a few minutes)", flush=True)
        time.sleep(poll)


def train(config_path: Path) -> None:
    import deeplabcut

    deeplabcut.train_network(str(config_path), shuffle=1, trainingsetindex=0)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("project", help="Project folder or its config.yaml.")
    parser.add_argument("--watch", action="store_true", help="Only watch a training that is already running.")
    parser.add_argument("--overwrite", action="store_true", help="Train although the train folder has snapshots.")
    parser.add_argument("--poll", type=float, default=10, help="Seconds between reads of train.txt (default 10).")
    parser.add_argument("--child", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()

    config_path = Path(args.project)
    if config_path.is_dir():
        config_path = config_path / "config.yaml"
    if args.child:
        train(config_path)
        return

    train_dir = find_train_dir(config_path)
    if args.watch:
        try:
            watch(train_dir, None, args.poll)
        except KeyboardInterrupt:
            print("stopped watching; training (if any) keeps running")
        return

    snapshots = sorted(train_dir.glob("snapshot-*.pt"))
    if snapshots and not args.overwrite:
        raise SystemExit(f"{train_dir} already has {len(snapshots)} snapshots; use --overwrite to train again")
    with open(train_dir / "pytorch_config.yaml", encoding="utf-8") as handle:
        settings = yaml.safe_load(handle)
    print(f"training {train_dir.parent.name}: {settings['train_settings']['epochs']} epochs, net {settings['net_type']}")
    print(f"console output: {train_dir / 'train_console.log'}")

    train_txt = train_dir / "train.txt"
    offset = train_txt.stat().st_size if train_txt.exists() else 0  # skip runs logged before this one
    # The child runs this file again (not -c) so dataloader workers can re-import it on Windows.
    with open(train_dir / "train_console.log", "a", encoding="utf-8") as log:
        log.write(f"\n=== {datetime.now():%Y-%m-%d %H:%M:%S} train_dlc_network.py {config_path}\n")  # noqa: DTZ005
        log.flush()
        process = subprocess.Popen([sys.executable, __file__, str(config_path), "--child"], stdout=log, stderr=subprocess.STDOUT)
        try:
            watch(train_dir, process, args.poll, offset)
            process.wait()
        except KeyboardInterrupt:
            print("stopping training...")
            process.terminate()
            process.wait()


if __name__ == "__main__":
    main()

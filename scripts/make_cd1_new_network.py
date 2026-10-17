"""
Creates a new DLC project (a new network) from the current iteration of an existing CD1 project, for a
different backbone. It has the same labeled frames and exactly the same train/test split, shuffle 1,
iteration 0, the source's augmentation, and a from-scratch 350-epoch schedule.

The new folder holds only what training and analysis read: config.yaml, the labeled PNGs listed in the
label table (hard-linked when possible) plus their CollectedData_GUSTAVO.h5, training-datasets and
dlc-models-pytorch. It is complete enough to zip and train on another machine. The source is not changed.

  python scripts\\make_cd1_new_network.py <source project> --date 2026-10-13 --net hrnet_w32
"""

import argparse
import os
import pickle
import shutil
from datetime import datetime
from pathlib import Path

import pandas as pd
import yaml

EPOCHS = 350


def link_or_copy(source: Path, target: Path) -> None:
    try:
        os.link(source, target)
    except OSError:
        shutil.copy2(source, target)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("source")
    parser.add_argument("--date", required=True, help="YYYY-MM-DD, used in the folder name and as the config date.")
    parser.add_argument("--net", default="hrnet_w32")
    args = parser.parse_args()

    source = Path(args.source)
    day = datetime.strptime(args.date, "%Y-%m-%d")  # noqa: DTZ007  (a calendar date, no time zone involved)
    dest = source.parent / f"CD1-GUSTAVO-{args.date}"
    if dest.exists():
        raise SystemExit(f"{dest} already exists")

    with open(source / "config.yaml", encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle)
    iteration = cfg["iteration"]
    source_data = source / "training-datasets" / f"iteration-{iteration}" / f"UnaugmentedDataSet_{cfg['Task']}{cfg['date']}"
    with open(source_data / f"Documentation_data-{cfg['Task']}_95shuffle1.pickle", "rb") as handle:
        _, train_indices, test_indices, _ = pickle.load(handle)
    table = pd.read_hdf(source_data / f"CollectedData_{cfg['scorer']}.h5")
    test_names = [tuple(table.index[i]) for i in test_indices]
    print(f"source iteration {iteration}: {len(train_indices)} train, {len(test_indices)} test of {len(table)} rows")

    dest.mkdir()
    folders = sorted({tuple(i)[1] for i in table.index})
    for folder in folders:
        target = dest / "labeled-data" / folder
        target.mkdir(parents=True)
        shutil.copy2(source / "labeled-data" / folder / f"CollectedData_{cfg['scorer']}.h5", target / f"CollectedData_{cfg['scorer']}.h5")
    for index in table.index:
        relative = Path(*[str(x) for x in index])
        link_or_copy(source / relative, dest / relative)

    new_cfg = dict(cfg)
    new_cfg["date"] = day.strftime("%b%d")
    new_cfg["iteration"] = 0
    new_cfg["project_path"] = str(dest)
    new_cfg["default_net_type"] = args.net
    with open(dest / "config.yaml", "w", encoding="utf-8") as handle:
        yaml.safe_dump(new_cfg, handle, sort_keys=False, allow_unicode=True)

    from deeplabcut.core.engine import Engine
    from deeplabcut.generate_training_dataset import create_training_dataset
    from deeplabcut.generate_training_dataset.trainingsetmanipulation import merge_annotateddatasets
    from deeplabcut.utils import auxiliaryfunctions

    config_path = str(dest / "config.yaml")
    dest_data = dest / "training-datasets" / "iteration-0" / f"UnaugmentedDataSet_{new_cfg['Task']}{new_cfg['date']}"
    dest_data.mkdir(parents=True)
    merged = merge_annotateddatasets(auxiliaryfunctions.read_config(config_path), dest_data)
    index_of = {tuple(i): n for n, i in enumerate(merged.index)}
    test = sorted(index_of[name] for name in test_names)
    train = [n for n in range(len(merged)) if n not in set(test)]
    print(f"new table {len(merged)} rows -> {len(train)} train, {len(test)} test")
    assert len(test) == len(test_names) == len(test_indices) and len(train) == len(train_indices)

    create_training_dataset(
        config_path,
        Shuffles=[1],
        trainIndices=[train],
        testIndices=[test],
        net_type=args.net,
        engine=Engine.PYTORCH,
        userfeedback=False,
    )

    old_cfg_path = next((source / "dlc-models-pytorch" / f"iteration-{iteration}").glob("*shuffle1")) / "train" / "pytorch_config.yaml"
    train_dir = next((dest / "dlc-models-pytorch" / "iteration-0").glob("*shuffle1")) / "train"
    new_cfg_path = train_dir / "pytorch_config.yaml"
    shutil.copy2(new_cfg_path, train_dir / "pytorch_config.yaml.orig")
    with open(old_cfg_path, encoding="utf-8") as handle:
        old = yaml.safe_load(handle)
    with open(new_cfg_path, encoding="utf-8") as handle:
        new = yaml.safe_load(handle)
    # Keep the new backbone's model section; take augmentation and training settings from the source.
    template_data = new["data"]
    new["data"] = old["data"]
    if args.net.startswith("hrnet"):
        # HRNet fuses branches at 1/4 to 1/32 resolution, so every input must be a multiple of 32 in both sides.
        # The template pads train and inference images for that; copying the source's data section drops it.
        for mode in ("train", "inference"):
            new["data"][mode]["auto_padding"] = template_data[mode]["auto_padding"]
        crop = new["data"]["train"].get("crop_sampling")
        if crop:
            crop["width"] = crop["height"] = (crop["width"] // 32) * 32
        print(f"hrnet: auto_padding {template_data['inference']['auto_padding']}, crop {crop and crop['width']}")
    new["train_settings"] = {**old["train_settings"], "epochs": EPOCHS}
    lr = new["runner"]["optimizer"]["params"]["lr"]
    new["runner"]["scheduler"] = {
        "type": "LRListScheduler",
        "params": {"lr_list": [[lr / 5], [lr / 25], [lr / 100]], "milestones": [200, 280, 320]},
    }
    new["runner"]["snapshots"] = {"max_snapshots": 8, "save_epochs": 25, "save_optimizer_state": False}
    with open(new_cfg_path, "w", encoding="utf-8") as handle:
        yaml.safe_dump(new, handle, sort_keys=False)

    shutil.copy2(Path(__file__).with_name("fix_project_paths.py"), dest / "fix_project_paths.py")

    # Verify: same test images by name, backbone as requested
    with open(dest_data / f"Documentation_data-{new_cfg['Task']}_95shuffle1.pickle", "rb") as handle:
        _, new_train, new_test, _ = pickle.load(handle)
    names = {tuple(merged.index[i]) for i in new_test}
    print("test images identical to source:", names == set(test_names), "| train", len(new_train), "test", len(new_test))
    print("net:", new["net_type"], "| optimizer:", new["runner"]["optimizer"], "| scheduler:", new["runner"]["scheduler"])
    size = sum(f.stat().st_size for f in dest.rglob("*") if f.is_file())
    print(f"project: {dest} | {len(folders)} folders, {len(table)} images, {size / 1e6:.0f} MB (hard-linked PNGs counted)")


if __name__ == "__main__":
    main()

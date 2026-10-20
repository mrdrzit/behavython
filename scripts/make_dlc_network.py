"""
Creates a new DLC project (a new network) from the current iteration of an existing single-animal PyTorch
project, for a different backbone or recipe. It has the labeled frames of the source's labeled-data folders listed in
video_sets, the same test images (the source's shuffle 1 split), shuffle 1, iteration 0, the source's augmentation
unless a recipe option changes it, and a from-scratch schedule. Frames labeled after the source split was built go to
train. Task, scorer and bodyparts come from the source's config.yaml. A TensorFlow (DLC 2.x) source also works:
augmentation and training settings then start from the PyTorch template.

The new folder, <Task>-<scorer>-<date>[-<suffix>], holds only what training and analysis read: config.yaml, the
labeled PNGs (hard-linked when possible) plus their CollectedData_<scorer>.h5,
training-datasets and dlc-models-pytorch. It is complete enough to zip and train on another machine. The source is
not changed.

  python scripts\\make_dlc_network.py <source project> --date 2026-10-13 --net hrnet_w32

--recipe tf builds the network that matched DLC 2.3 TensorFlow on the CD1 videos (CD1-GUSTAVO-2026-10-15):
resnet_50 with the SuperAnimal-TopViewMouse backbone, whole frames resized to about TensorFlow's global_scale 0.8,
TensorFlow's plateau score map with sigmoid cross-entropy, no flips, light affine augmentation, 300 epochs. Options
given explicitly override the recipe (for animals other than top-view mice, add --superanimal none or
--superanimal superanimal_quadruped).

  python scripts\\make_dlc_network.py <source project> --date 2026-11-02 --recipe tf ^
      --dest-root E:\\Matheus\\Deeplabcut_networks
"""

import argparse
import os
import pickle
import shutil
import statistics
from datetime import datetime
from pathlib import Path

import pandas as pd
import yaml
from PIL import Image

EPOCHS = 350
MILESTONE_SHARES = (0.57, 0.8, 0.91)  # 200, 280, 320 of 350 epochs; 171, 240, 273 of 300
# TensorFlow recipe: whole frames at global_scale 0.8, scale jitter 0.5 to 1.25, rotation 25 on 40% of the
# images, no noise, blur or covering. The resize side is 0.8 of the median longest frame side, as a multiple of 32
# (576 for the CD1 frames, which are 696 to 740 px).
TF_GLOBAL_SCALE = 0.8
SIZE_DIVISOR = 32
SUPERANIMALS = ("none", "superanimal_topviewmouse", "superanimal_quadruped")
RECIPES = {
    "tf": {
        "net": "resnet_50",
        "epochs": 300,
        "fullframe": True,
        "head": "plateau-bce",
        "no_hflip": True,
        "superanimal": "superanimal_topviewmouse",
    },
}


def link_or_copy(source: Path, target: Path) -> None:
    try:
        os.link(source, target)
    except OSError:
        shutil.copy2(source, target)


def split_file(data_folder: Path, cfg: dict) -> Path:
    """Same name as deeplabcut.utils.auxiliaryfunctions.get_data_and_metadata_filenames for shuffle 1."""
    fraction = int(100 * cfg["TrainingFraction"][0])
    return data_folder / f"Documentation_data-{cfg['Task']}_{fraction}shuffle1.pickle"


def fullframe_size(project: Path, folders: list[str]) -> int:
    """TensorFlow's global_scale applied to the median longest side of the labeled frames, as a multiple of 32."""
    sides = []
    for folder in folders:
        image = next((project / "labeled-data" / folder).glob("*.png"), None)
        if image is not None:
            with Image.open(image) as handle:
                sides.append(max(handle.size))
    return max(SIZE_DIVISOR, round(statistics.median(sides) * TF_GLOBAL_SCALE / SIZE_DIVISOR) * SIZE_DIVISOR)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("source")
    parser.add_argument("--date", required=True, help="YYYY-MM-DD, used in the folder name and as the config date.")
    parser.add_argument("--recipe", choices=sorted(RECIPES), help="Preset of the options below; explicit options win.")
    parser.add_argument("--dest-root", help="Folder that receives the new project (default: the source's parent).")
    parser.add_argument("--net", default="hrnet_w32")
    parser.add_argument("--suffix", default="", help="Appended to the folder name, e.g. fullframe. DLC names still use Task and date.")
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--fullframe", action="store_true", help="Whole frames resized to the TensorFlow scale instead of crops.")
    parser.add_argument("--resize", type=int, help="Side of the --fullframe resize (default: from the labeled frames).")
    parser.add_argument(
        "--head",
        choices=("gaussian", "plateau-bce"),
        default="gaussian",
        help="plateau-bce: TensorFlow's score map (disk of pos_dist_thresh, sigmoid cross-entropy, sigmoid output).",
    )
    parser.add_argument("--no-hflip", action="store_true", help="No horizontal flips (TensorFlow mirror: false).")
    parser.add_argument(
        "--superanimal",
        choices=SUPERANIMALS,
        default="none",
        help="Backbone from a SuperAnimal model (transfer learning: no head weights, no detector).",
    )
    known, _ = parser.parse_known_args()
    if known.recipe:
        parser.set_defaults(**RECIPES[known.recipe])
    return parser.parse_args()


def main():
    args = parse_args()

    source = Path(args.source)
    day = datetime.strptime(args.date, "%Y-%m-%d")  # noqa: DTZ007  (a calendar date, no time zone involved)
    with open(source / "config.yaml", encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle)
    if cfg.get("multianimalproject"):
        raise SystemExit("multi-animal projects are not supported")
    iteration = cfg["iteration"]
    source_models = source / "dlc-models-pytorch" / f"iteration-{iteration}"
    source_pytorch_cfg = next(source_models.glob("*shuffle1/train/pytorch_config.yaml"), None)
    if source_pytorch_cfg is None:
        print(f"{source_models} has no shuffle 1 pytorch_config.yaml: TensorFlow source, using the template's settings")
    root = Path(args.dest_root) if args.dest_root else source.parent
    dest = root / (f"{cfg['Task']}-{cfg['scorer']}-{args.date}" + (f"-{args.suffix}" if args.suffix else ""))
    if dest.exists():
        raise SystemExit(f"{dest} already exists")

    source_data = source / "training-datasets" / f"iteration-{iteration}" / f"UnaugmentedDataSet_{cfg['Task']}{cfg['date']}"
    with open(split_file(source_data, cfg), "rb") as handle:
        _, train_indices, test_indices, _ = pickle.load(handle)
    split_table = pd.read_hdf(source_data / f"CollectedData_{cfg['scorer']}.h5")
    test_names = [tuple(split_table.index[i]) for i in test_indices]
    print(f"source iteration {iteration}: {len(train_indices)} train, {len(test_indices)} test of {len(split_table)} rows")

    from deeplabcut.utils.conversioncode import guarantee_multiindex_rows

    # Same folders merge_annotateddatasets reads: labeled-data/<video stem> for each video_sets entry.
    tables = []
    for stem in sorted({Path(video.replace("\\", "/")).stem for video in cfg["video_sets"]}):
        label_file = source / "labeled-data" / stem / f"CollectedData_{cfg['scorer']}.h5"
        if label_file.exists():
            frames = pd.read_hdf(label_file)
            guarantee_multiindex_rows(frames)
            tables.append(frames)
    table = pd.concat(tables)
    missing = set(test_names) - {tuple(i) for i in table.index}
    if missing:
        raise SystemExit(f"{len(missing)} test images of the source split are not in labeled-data, e.g. {min(missing)}")
    print(f"labeled-data: {len(table)} frames, {len(table) - len(split_table)} more than the source split")

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
    # create_training_dataset names the split after round(train / total, 2); config must match or training cannot find it
    new_cfg["TrainingFraction"] = [round((len(table) - len(test_names)) / len(table), 2)]
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
    assert len(test) == len(test_names) == len(test_indices) and len(merged) == len(table)

    weight_init = None
    if args.superanimal != "none":
        from deeplabcut.modelzoo import build_weight_init

        weight_init = build_weight_init(new_cfg, super_animal=args.superanimal, model_name=args.net, detector_name=None)
        print(f"{args.superanimal} backbone:", weight_init.snapshot_path)

    create_training_dataset(
        config_path,
        Shuffles=[1],
        trainIndices=[train],
        testIndices=[test],
        net_type=args.net,
        engine=Engine.PYTORCH,
        userfeedback=False,
        weight_init=weight_init,
    )

    train_dir = next((dest / "dlc-models-pytorch" / "iteration-0").glob("*shuffle1")) / "train"
    new_cfg_path = train_dir / "pytorch_config.yaml"
    shutil.copy2(new_cfg_path, train_dir / "pytorch_config.yaml.orig")
    with open(new_cfg_path, encoding="utf-8") as handle:
        new = yaml.safe_load(handle)
    old = new  # TensorFlow source: nothing to carry over, the template's data and train_settings stay
    if source_pytorch_cfg is not None:
        with open(source_pytorch_cfg, encoding="utf-8") as handle:
            old = yaml.safe_load(handle)
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
    if args.fullframe:
        size = args.resize or fullframe_size(source, folders)
        for mode in ("train", "inference"):
            section = new["data"][mode]
            section["resize"] = {"height": size, "width": size, "keep_ratio": True}
        train = new["data"]["train"]
        train["crop_sampling"] = None
        train["affine"] = {"p": 0.4, "rotation": 25, "scaling": [0.5, 1.25], "translation": 0}
        for key in ("hist_eq", "motion_blur", "covering", "gaussian_noise"):
            train[key] = False
        print(f"fullframe: resize {size}, no crop_sampling, affine {train['affine']}")
    if args.no_hflip:
        new["data"]["train"]["hflip"] = False
    if args.head == "plateau-bce":
        head = new["model"]["heads"]["bodypart"]
        head["target_generator"]["type"] = "HeatmapPlateauGenerator"
        head["criterion"]["heatmap"]["type"] = "WeightedBCECriterion"
        head["predictor"]["apply_sigmoid"] = True
        head["predictor"]["clip_scores"] = False
        print("head: plateau target, BCE loss, sigmoid output")
    # The source's train_settings would drop the weight_init that create_training_dataset wrote.
    weight_init_cfg = new["train_settings"].get("weight_init")
    new["train_settings"] = {**old["train_settings"], "epochs": args.epochs, "weight_init": weight_init_cfg}
    lr = new["runner"]["optimizer"]["params"]["lr"]
    new["runner"]["scheduler"] = {
        "type": "LRListScheduler",
        "params": {"lr_list": [[lr / 5], [lr / 25], [lr / 100]], "milestones": [round(args.epochs * share) for share in MILESTONE_SHARES]},
    }
    new["runner"]["snapshots"] = {"max_snapshots": 8, "save_epochs": 25, "save_optimizer_state": False}
    # snapshot-best follows the lowest test RMSE: test mAP saturates at 100 on small test sets and then never improves
    new["runner"]["key_metric"] = "test.rmse"
    new["runner"]["key_metric_asc"] = False
    with open(new_cfg_path, "w", encoding="utf-8") as handle:
        yaml.safe_dump(new, handle, sort_keys=False)

    shutil.copy2(Path(__file__).with_name("fix_project_paths.py"), dest / "fix_project_paths.py")

    # Verify: same test images by name, backbone as requested
    with open(split_file(dest_data, new_cfg), "rb") as handle:
        _, new_train, new_test, _ = pickle.load(handle)
    names = {tuple(merged.index[i]) for i in new_test}
    print("test images identical to source:", names == set(test_names), "| train", len(new_train), "test", len(new_test))
    print("net:", new["net_type"], "| optimizer:", new["runner"]["optimizer"], "| scheduler:", new["runner"]["scheduler"])
    size = sum(f.stat().st_size for f in dest.rglob("*") if f.is_file())
    print(f"project: {dest} | {len(folders)} folders, {len(table)} images, {size / 1e6:.0f} MB (hard-linked PNGs counted)")


if __name__ == "__main__":
    main()

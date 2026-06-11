import argparse
from pathlib import Path
from itertools import islice
import pandas as pd
import yaml


def guarantee_multiindex_rows(df: pd.DataFrame):
    """Standard DLC logic to ensure index is a platform-agnostic MultiIndex."""
    if not isinstance(df.index, pd.MultiIndex):
        path = df.index[0]
        try:
            sep = "/" if "/" in path else "\\"
            splits = tuple(df.index.str.split(sep))
            df.index = pd.MultiIndex.from_tuples(splits)
        except (TypeError, AttributeError):
            pass

    try:
        df.index = df.index.set_levels(df.index.levels[1].astype(str), level=1)
    except (AttributeError, IndexError):
        pass


def synchronize_csv_to_h5(csv_path: Path, config_path: str = None):
    """
    Reads a DLC-style CSV file, ensures its headers and index are formatted correctly,
    and saves both the cleaned CSV and its required H5 counterpart.
    """
    # 1. Determine the scorer
    scorer = None
    if config_path:
        config_file = Path(config_path)
        if config_file.exists():
            with open(config_file, "r") as f:
                config_data = yaml.safe_load(f)
                scorer = config_data.get("scorer")
        else:
            print(f"  Warning: Config file not found at {config_path}. Will attempt to infer scorer from CSV.")

    # 2. Determine header depth without loading the whole file
    with open(csv_path) as datafile:
        head = list(islice(datafile, 0, 5))

    if not head:
        print(f"  Error: {csv_path.name} is empty.")
        return

    # Check for 'individuals' to see if it's a multi-animal project (4-row header)
    if len(head) > 1 and "individuals" in head[1]:
        header = list(range(4))
    else:
        header = list(range(3))

    # Check if the first column is 'labeled-data' for index cols
    if head[-1].split(",")[0] == "labeled-data":
        index_col = [0, 1, 2]
    else:
        index_col = 0

    # 3. Read CSV with exact DLC parameters
    try:
        data = pd.read_csv(csv_path, index_col=index_col, header=header)
    except Exception as e:
        print(f"  Error loading CSV {csv_path.name}: {e}")
        return

    # 4. Enforce scorer if inferred or provided
    if not scorer:
        try:
            scorer = data.columns.get_level_values("scorer")[0]
        except (KeyError, IndexError, ValueError):
            try:
                scorer = data.columns.get_level_values(0)[0]
            except Exception:
                scorer = "unknown_scorer"
                print("  Warning: Could not infer scorer from CSV. Using 'unknown_scorer'.")

    try:
        # First try using level name
        data.columns = data.columns.set_levels([scorer], level="scorer")
    except (KeyError, ValueError):
        # Fallback to level index 0
        try:
            data.columns = data.columns.set_levels([scorer], level=0)
        except Exception as e:
            print(f"  Warning: Could not set scorer level: {e}")

    # 5. Apply platform-agnostic index formatting
    guarantee_multiindex_rows(data)

    # 6. Save to H5 and back to CSV to ensure consistency
    h5_path = csv_path.with_suffix(".h5")
    
    try:
        # Save H5 with format fixed, required for DLC CollectedData
        data.to_hdf(h5_path, key="df_with_missing", mode="w", format="fixed")
        data.to_csv(csv_path)
        print(f"  -> Successfully synchronized {csv_path.name} to {h5_path.name} (Scorer: {scorer})")
    except Exception as e:
        print(f"  Error saving files for {csv_path.name}: {e}")


def process_folders(folders: list[str], config_path: str = None):
    """Iterates through folders and syncs any CollectedData CSVs found."""
    for folder_str in folders:
        print("-" * 50)
        folder_path = Path(folder_str)
        if not folder_path.exists() or not folder_path.is_dir():
            print(f"Error: Directory does not exist or is invalid -> {folder_path}")
            continue

        print(f"Processing folder: {folder_path.name}")
        csv_files = list(folder_path.glob("CollectedData_*.csv"))
        
        if not csv_files:
            print(f"  Warning: No CollectedData_*.csv found in {folder_path.name}. Skipping.")
            continue

        for csv_file in csv_files:
            synchronize_csv_to_h5(csv_file, config_path)
    print("-" * 50)


if __name__ == "__main__":
    description = (
        "Synchronize manual DLC annotations (CSV) into platform-agnostic MultiIndex H5 files.\n\n"
        "This utility acts as a standalone version of DeepLabCut's convertcsv2h5 function.\n"
        "It reads raw CSV annotation files, enforces the correct header/index structure,\n"
        "and generates the H5 file required for DLC compatibility."
    )

    parser = argparse.ArgumentParser(description=description, formatter_class=argparse.RawTextHelpFormatter)

    parser.add_argument(
        "folders",
        nargs="+",
        help="One or more directory paths containing CollectedData_*.csv files."
    )

    parser.add_argument(
        "-c",
        "--config",
        help="Optional path to a target config.yaml file to enforce the correct scorer name.",
        default=None,
    )

    args = parser.parse_args()
    process_folders(args.folders, args.config)

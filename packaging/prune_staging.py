import os
import shutil
from pathlib import Path

# Resolve staging directory
staging_dir = Path(r"build/staging_env").resolve()
site_packages = staging_dir / "Lib" / "site-packages"

print(f"Scanning staging environment at: {staging_dir}")
print(f"Scanning site-packages at: {site_packages}")

test_dirs = []
for d in site_packages.rglob("*"):
    if not d.is_dir():
        continue
    if d.name.lower() in ("tests", "test", "testing"):
        rel_parts = d.relative_to(site_packages).parts
        # Protect behavython and deeplabcut
        if "behavython" in rel_parts or "deeplabcut" in rel_parts:
            continue
        test_dirs.append(d)

print(f"Found {len(test_dirs)} third-party test directories to prune.")

deleted_files = 0
deleted_dirs = 0

for d in test_dirs:
    if d.exists() and d.is_dir():
        file_count = sum(len(files) for _, _, files in os.walk(d))
        try:
            shutil.rmtree(d)
            deleted_dirs += 1
            deleted_files += file_count
        except Exception as e:
            print(f"Error removing {d}: {e}")

print(f"Successfully pruned {deleted_dirs} test directories containing {deleted_files} files.")

# Remove .pdb debug symbol files
pdb_count = 0
for p in staging_dir.rglob("*.pdb"):
    try:
        p.unlink()
        pdb_count += 1
    except Exception as e:
        print(f"Error removing {p}: {e}")
print(f"Removed {pdb_count} .pdb files.")

# Remove development include and Tools directories
for folder_name in ["include", "Tools"]:
    p = staging_dir / folder_name
    if p.exists() and p.is_dir():
        cnt = sum(len(f) for _, _, f in os.walk(p))
        try:
            shutil.rmtree(p)
            deleted_files += cnt
            print(f"Removed {folder_name} ({cnt} files).")
        except Exception as e:
            print(f"Error deleting {p}: {e}")

total_remaining = sum(len(files) for _, _, files in os.walk(staging_dir))
print(f"Total remaining files in staging: {total_remaining}")

import os
from pathlib import Path

orig = Path(r"C:\Users\uzuna\miniforge3\envs\behavython")
stg = Path(r"build\staging_env").resolve()

print("=== 1. Specific Folder Verification ===")
print("staging/include exists:", (stg / "include").exists())
print("staging/Tools exists:  ", (stg / "Tools").exists())
print("staging/python.pdb exists:", (stg / "python.pdb").exists())

print("\n=== 2. Sample Third-Party Test Folders ===")
sample_tests = ["bottleneck/tests", "colorama/tests", "certifi/tests", "adodbapi/test"]
for s in sample_tests:
    orig_path = orig / "Lib" / "site-packages" / Path(s)
    stg_path = stg / "Lib" / "site-packages" / Path(s)
    print(f"  {s:<22}: original={orig_path.exists()}  |  staging={stg_path.exists()}")

print("\n=== 3. Protected Packages Check ===")
for p in ["behavython", "deeplabcut"]:
    p_path = stg / "Lib" / "site-packages" / p
    print(f"  {p:<22}: exists={p_path.exists()}")

print("\n=== 4. File Count Comparison ===")
orig_files = sum(len(f) for _, _, f in os.walk(orig))
stg_files = sum(len(f) for _, _, f in os.walk(stg))
print(f"  Original conda env: {orig_files} files")
print(f"  Staging directory:  {stg_files} files")
print(f"  Total pruned:       {orig_files - stg_files} files removed")

"""Regression test: analysis outputs must match the saved reference in tests/expected.

Run:     python -m pytest
Update:  python -m pytest --update-expected   (only after an intended change to results)
"""

import os
import shutil
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest

TESTS_DIR = Path(__file__).parent
REPO_ROOT = TESTS_DIR.parent
EXPECTED_DIR = TESTS_DIR / "expected"

DATASETS = {
    "open_field": "open_field",
    "plus_maze": "elevated_plus_maze",
    "social_discrimination": "social_discrimination",
    "social_recognition": "social_recognition",
}


def run_analysis(dataset: str, output_dir: Path) -> None:
    input_dir = TESTS_DIR / dataset
    cmd = [sys.executable, "-m", "behavython.cli", "-f", str(input_dir), "-o", str(output_dir)]
    cmd += ["--experiment-type", DATASETS[dataset], "--no-plots"]
    arena = input_dir / "arena.json"
    if arena.exists():
        cmd += ["--arena-config", str(arena)]

    env = {**os.environ, "PYTHONPATH": str(REPO_ROOT / "src")}
    result = subprocess.run(cmd, capture_output=True, text=True, env=env, check=False)
    # 2 means the run finished but some animals were invalid, which these datasets contain on purpose.
    assert result.returncode in (0, 2), f"CLI failed with exit code {result.returncode}:\n{result.stdout[-3000:]}\n{result.stderr[-3000:]}"


@pytest.mark.parametrize("dataset", DATASETS)
def test_analysis_outputs_match_expected(dataset, tmp_path, request):
    run_analysis(dataset, tmp_path)
    produced = {p.name: p for p in tmp_path.glob("*.parquet")}
    expected_dir = EXPECTED_DIR / dataset

    if request.config.getoption("--update-expected"):
        shutil.rmtree(expected_dir, ignore_errors=True)
        expected_dir.mkdir(parents=True)
        for path in produced.values():
            shutil.copy2(path, expected_dir / path.name)
        pytest.skip(f"Updated {len(produced)} reference files in {expected_dir}")

    assert expected_dir.exists(), f"No reference outputs in {expected_dir}. Run: python -m pytest --update-expected"
    expected = {p.name: p for p in expected_dir.glob("*.parquet")}
    assert sorted(produced) == sorted(expected), "The set of output files changed"

    for name in sorted(expected):
        try:
            pd.testing.assert_frame_equal(pd.read_parquet(produced[name]), pd.read_parquet(expected[name]), check_exact=False, rtol=1e-9)
        except AssertionError as e:
            raise AssertionError(f"{dataset}/{name} differs from the reference:\n{e}") from None

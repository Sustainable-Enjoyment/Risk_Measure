import sys
import os
from pathlib import Path

import pytest

# Ensure the repository root is on the import path
sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.risk_measure import tokenize, compute_score


def test_tokenize_basic():
    text = "Supply-chain risks, e.g., delays? 123"
    tokens = tokenize(text)
    assert tokens == ["supply", "chain", "risks", "e", "g", "delays"]


def test_compute_score_positive():
    tokens = ["delay", "supply", "risk"]
    sc_words = {"supply"}
    risk_words = {"risk"}
    score = compute_score(tokens, sc_words, risk_words, window=1)
    assert score == pytest.approx(1 / 3)


def test_compute_score_zero():
    tokens = ["supply", "a", "a", "risk"]
    sc_words = {"supply"}
    risk_words = {"risk"}
    score = compute_score(tokens, sc_words, risk_words, window=1)
    assert score == 0


def test_run_measure_cli(tmp_path):
    """run_measure should produce an Excel file with SCRisk_Score column."""
    import subprocess
    import pandas as pd

    repo_root = Path(__file__).resolve().parents[1]
    excel_path = repo_root / "examples" / "EarningCall_demo.xlsx"
    sample_df = pd.read_excel(excel_path).head(2)
    sample_file = tmp_path / "sample.xlsx"
    sample_df.to_excel(sample_file, index=False)

    output_file = tmp_path / "scores.xlsx"
    cmd = [
        sys.executable,
        str(repo_root / "scripts" / "run_measure.py"),
        str(sample_file),
        "--output",
        str(output_file),
    ]
    env = dict(os.environ)
    env["PYTHONPATH"] = str(repo_root / "src")
    subprocess.run(cmd, check=True, env=env, cwd=repo_root)

    assert output_file.exists()
    result_df = pd.read_excel(output_file)
    assert "SCRisk_Score" in result_df.columns

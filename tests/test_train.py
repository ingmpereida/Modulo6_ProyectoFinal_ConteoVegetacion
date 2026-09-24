"""Tests for train.py pure seams and CLI contract (FR-5, NFR-4).

train.py must stay importable WITHOUT the ultralytics runtime so the pytest
suite stays CPU-only (FR-6) — the heavy import is lazy, inside run(). These
tests cover only the pure seams (kwargs assembly, metric extraction/format)
and the CLI contract via subprocess.
"""

import subprocess
import sys
from pathlib import Path

import pytest

from train import build_train_kwargs, extract_metrics, format_metrics

REPO_ROOT = Path(__file__).resolve().parent.parent


# ---------------------------------------------------------------------------
# FR-5: build_train_kwargs — assemble the ultralytics train() overrides
# ---------------------------------------------------------------------------


class TestBuildTrainKwargs:
    def test_full_overrides_surface(self):
        kwargs = build_train_kwargs(
            data="/tmp/ds/data.yaml",
            epochs=100,
            imgsz=640,
            batch=8,
            project="runs/",
            resume="runs/train/exp/weights/last.pt",
        )

        assert kwargs == {
            "data": "/tmp/ds/data.yaml",
            "epochs": 100,
            "imgsz": 640,
            "batch": 8,
            "project": "runs/",
            "resume": "runs/train/exp/weights/last.pt",
        }

    def test_resume_omitted_when_not_resuming(self):
        kwargs = build_train_kwargs(
            data="d.yaml", epochs=1, imgsz=640, batch=2, project="runs/"
        )

        assert "resume" not in kwargs
        assert kwargs["data"] == "d.yaml"


# ---------------------------------------------------------------------------
# FR-5: extract_metrics — pull mAP50 / mAP50-95 from model.val() (F-1 gate:
# verified on ultralytics 8.4.161, DetMetrics.results_dict keys)
# ---------------------------------------------------------------------------


class TestExtractMetrics:
    def test_reads_map_keys_from_results_dict(self):
        stub = type(
            "Val",
            (),
            {"results_dict": {"metrics/mAP50(B)": 0.5, "metrics/mAP50-95(B)": 0.25}},
        )()

        assert extract_metrics(stub) == {"mAP50": 0.5, "mAP50-95": 0.25}

    def test_coerces_tensor_values_via_item(self):
        class Tensor:
            def __init__(self, value):
                self.value = value

            def item(self):
                return self.value

        stub = type(
            "Val",
            (),
            {
                "results_dict": {
                    "metrics/mAP50(B)": Tensor(0.75),
                    "metrics/mAP50-95(B)": Tensor(0.4),
                }
            },
        )()

        assert extract_metrics(stub) == {"mAP50": 0.75, "mAP50-95": 0.4}

    def test_missing_map_key_raises_fail_fast(self):
        stub = type("Val", (), {"results_dict": {"metrics/precision(B)": 0.9}})()

        with pytest.raises(ValueError, match="metrics/mAP50"):
            extract_metrics(stub)


# ---------------------------------------------------------------------------
# FR-5: format_metrics — readable mAP line printed after validation
# ---------------------------------------------------------------------------


class TestFormatMetrics:
    def test_three_decimal_format(self):
        line = format_metrics({"mAP50": 0.5000, "mAP50-95": 0.2500})

        assert line == "Validation metrics - mAP50 (B): 0.500 | mAP50-95 (B): 0.250"

    def test_rounds_and_labels_both_metrics(self):
        line = format_metrics({"mAP50": 0.123456, "mAP50-95": 0.098765})

        assert "mAP50 (B): 0.123" in line
        assert "mAP50-95 (B): 0.099" in line


# ---------------------------------------------------------------------------
# FR-5: CLI contract (subprocess; must work WITHOUT ultralytics installed)
# ---------------------------------------------------------------------------


class TestTrainCli:
    def test_help_exits_zero_and_lists_fr5_args(self):
        proc = subprocess.run(
            [sys.executable, "train.py", "--help"],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
        )

        assert proc.returncode == 0
        for arg in ("--data", "--weights", "--epochs", "--imgsz", "--batch", "--project", "--resume"):
            assert arg in proc.stdout

    def test_missing_required_data_exits_two(self):
        proc = subprocess.run(
            [sys.executable, "train.py"],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
        )

        assert proc.returncode == 2
        assert "--data" in proc.stderr
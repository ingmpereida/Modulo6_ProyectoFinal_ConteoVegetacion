#!/usr/bin/env python3
"""train.py — train a YOLO26 detection model on a prepared dataset (FR-5).

Consumes the dataset layout written by prepare_dataset.py (its data.yaml) and
wraps the pinned ultralytics runtime: YOLO(weights) -> model.train(...) ->
model.val() -> mAP50 / mAP50-95 printed to stdout. All artifacts land under
the gitignored --project root (runs/ by default).

The ultralytics import is deliberately lazy (inside run()): --help and the
pytest suite work without the heavy runtime (FR-6, NFR-4).
"""

import argparse
import sys

DEFAULT_WEIGHTS = "yolo26n.pt"
DEFAULT_EPOCHS = 100
DEFAULT_IMGSZ = 640
DEFAULT_BATCH = 8
DEFAULT_PROJECT = "runs/"
# F-1 gate (2026-09-23, ultralytics 8.4.161): model.val() returns DetMetrics
# whose results_dict carries exactly these keys.
MAPPING_KEYS = ("metrics/mAP50(B)", "metrics/mAP50-95(B)")


def build_train_kwargs(
    data: str,
    epochs: int,
    imgsz: int,
    batch: int,
    project: str,
    resume: str | None = None,
) -> dict:
    """Assemble the overrides passed to model.train(...) (FR-5).

    Only values the user actually set are emitted; resume is omitted unless a
    checkpoint path was given, so a fresh run trains from --weights.
    """
    kwargs = {
        "data": data,
        "epochs": epochs,
        "imgsz": imgsz,
        "batch": batch,
        "project": project,
    }
    if resume:
        kwargs["resume"] = resume
    return kwargs


def extract_metrics(metrics) -> dict[str, float]:
    """Pull mAP50 / mAP50-95 out of a model.val() result (FR-5).

    Verified on the pinned ultralytics 8.4.161: val() returns a DetMetrics
    instance whose results_dict exposes "metrics/mAP50(B)" and
    "metrics/mAP50-95(B)". Fail fast on drift instead of printing garbage.
    """
    if not hasattr(metrics, "results_dict"):
        raise ValueError(
            f"unexpected val() result type {type(metrics).__name__}: no results_dict"
        )
    results = metrics.results_dict
    extracted = {}
    for key, label in ((MAPPING_KEYS[0], "mAP50"), (MAPPING_KEYS[1], "mAP50-95")):
        if key not in results:
            raise ValueError(f"val() result missing {key!r}; got keys {sorted(results)}")
        value = results[key]
        extracted[label] = float(value.item() if hasattr(value, "item") else value)
    return extracted


def format_metrics(metrics: dict[str, float]) -> str:
    """Render the validation summary line printed to the user (FR-5)."""
    return (
        f"Validation metrics - mAP50 (B): {metrics['mAP50']:.3f} "
        f"| mAP50-95 (B): {metrics['mAP50-95']:.3f}"
    )


def run(args: argparse.Namespace) -> None:
    """Train on a prepared dataset and print validation mAP (FR-5)."""
    from ultralytics import YOLO  # lazy: keeps --help and tests ultralytics-free

    if args.resume:
        model = YOLO(args.resume)  # load the checkpoint directly, no fresh download
    else:
        model = YOLO(args.weights)

    overrides = build_train_kwargs(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        project=args.project,
        resume=args.resume,
    )
    model.train(**overrides)
    results = model.val()
    print(format_metrics(extract_metrics(results)))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="train.py",
        description="Train a YOLO26 detection model on a prepare_dataset.py dataset (FR-5).",
    )
    parser.add_argument(
        "--data", required=True, help="path to the data.yaml prepared by prepare_dataset.py"
    )
    parser.add_argument(
        "--weights",
        default=DEFAULT_WEIGHTS,
        help="pretrained weights (.pt) or model yaml (default: yolo26n.pt)",
    )
    parser.add_argument(
        "--epochs", type=int, default=DEFAULT_EPOCHS, help="training epochs (default: 100)"
    )
    parser.add_argument(
        "--imgsz", type=int, default=DEFAULT_IMGSZ, help="training image size (default: 640)"
    )
    parser.add_argument(
        "--batch", type=int, default=DEFAULT_BATCH, help="batch size (default: 8)"
    )
    parser.add_argument(
        "--project",
        default=DEFAULT_PROJECT,
        help="root for training artifacts under runs/ (default: runs/)",
    )
    parser.add_argument(
        "--resume",
        default=None,
        help="path to a last.pt checkpoint to continue from (default: none)",
    )
    args = parser.parse_args(argv)
    run(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
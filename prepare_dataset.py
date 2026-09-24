#!/usr/bin/env python3
"""prepare_dataset.py — build a YOLO dataset from tile_pipeline.py output.

Reads tile_pipeline.py's manifest.csv plus the per-flight tile folders and
YOLO label sidecars, and writes a standard dataset layout:

    <output>/
        images/{train,valid}/   tile images (copied, never modified)
        labels/{train,valid}/   YOLO .txt sidecars (only for labeled tiles)
        data.yaml               deterministic YAML pointing at the layout

Flights are split independently (train/valid, default 80/20, fixed seed) so
tiles never leak across flights (FR-2). A tile without a label sidecar becomes
a background sample: its image goes to images/ only, never to labels/ (FR-3).

The pipeline validates ALL inputs (manifest, image files, sidecar contents)
before creating --output; on any error nothing is written (FR-4).
"""

import argparse
import csv
import random
import shutil
import sys
from pathlib import Path

import yaml

DEFAULT_VALID_RATIO = 0.2
DEFAULT_SEED = 42
DEFAULT_NAMES = ["plant"]
YOLO_LINE_FIELDS = 5


class DatasetError(Exception):
    """User-facing error: invalid input or inconsistent dataset layout."""


class LabelFormatError(ValueError):
    """A label sidecar failed the YOLO 5-field-per-line parse (C-3)."""


def group_flights(rows: list[dict]) -> dict[str, list[dict]]:
    """Group manifest rows by flight (the vuelo column), preserving manifest order.

    FR-1: each manifest row is assigned to exactly one flight group.
    """
    groups: dict[str, list[dict]] = {}
    for row in rows:
        groups.setdefault(row["vuelo"], []).append(row)
    return groups


def main(argv: list[str] | None = None) -> int:
    raise NotImplementedError  # placeholder; CLI lands with tasks 2.11-2.12


if __name__ == "__main__":
    sys.exit(main())
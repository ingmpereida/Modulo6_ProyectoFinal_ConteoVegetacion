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


def split_tiles(
    tiles: list[str],
    valid_ratio: float = DEFAULT_VALID_RATIO,
    seed: int = DEFAULT_SEED,
) -> tuple[list[str], list[str]]:
    """Split one flight's tiles into (train, valid) deterministically.

    FR-2: per-flight split, fixed seed (NFR-1). n == 1 → valid is empty and
    the tile goes to train; otherwise valid = clamp(round(valid_ratio * n),
    1, n - 1). The input order never matters: the tile names are sorted and a
    seeded Random drives the shuffle, so identical inputs + seed always
    produce identical sets.
    """
    n = len(tiles)
    if n == 1:
        return list(tiles), []
    valid_count = max(1, min(n - 1, round(valid_ratio * n)))
    ordered = sorted(tiles)
    rng = random.Random(seed)
    rng.shuffle(ordered)
    return ordered[valid_count:], ordered[:valid_count]


def build_yaml(dataset_path: Path, names: list[str] | None = None) -> dict:
    """Return the data.yaml dict for a dataset layout (FR-3).

    Key order is fixed (path, train, val, names) so the CLI can emit it with
    yaml.safe_dump(sort_keys=False) and produce byte-identical files across
    runs (NFR-1).
    """
    return {
        "path": str(dataset_path),
        "train": "images/train",
        "val": "images/valid",
        "names": list(names if names is not None else DEFAULT_NAMES),
    }


def validate_label_file(sidecar: Path) -> None:
    """Validate a YOLO sidecar before it is copied (C-3).

    Contract: one box per line as `class x y w h` (normalized floats), so
    every non-empty line must have exactly 5 whitespace-separated fields, a
    non-negative integer class id, and 4 float coordinates. Raises
    LabelFormatError on malformed content — fail fast, never ship a silently
    corrupted dataset.
    """
    try:
        lines = sidecar.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise LabelFormatError(f"cannot read label file {sidecar}: {exc}") from exc

    for lineno, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        fields = line.split()
        if len(fields) != YOLO_LINE_FIELDS:
            raise LabelFormatError(
                f"malformed label {sidecar}:{lineno}: expected {YOLO_LINE_FIELDS} "
                f"fields (class x y w h), got {len(fields)}"
            )
        if not fields[0].isdigit():
            raise LabelFormatError(
                f"malformed label {sidecar}:{lineno}: class id {fields[0]!r} "
                f"is not a non-negative integer"
            )
        for field in fields[1:]:
            try:
                float(field)
            except ValueError:
                raise LabelFormatError(
                    f"malformed label {sidecar}:{lineno}: {field!r} is not a float"
                ) from None


def copy_label(src_txt: Path, dst_dir: Path) -> Path | None:
    """Copy a YOLO sidecar into dst_dir (FR-3).

    This is the single seam that reads label bytes (design D-3). Returns the
    destination path, or None when src_txt is missing — the tile is then a
    background sample and only its image is written. Malformed sidecars raise
    LabelFormatError so a corrupt dataset never ships silently.
    """
    if not src_txt.exists():
        return None
    validate_label_file(src_txt)
    dst = dst_dir / src_txt.name
    shutil.copyfile(src_txt, dst)
    return dst


def read_manifest(manifest_path: Path) -> list[dict]:
    """Parse manifest.csv into row dicts, validating its shape (FR-1).

    The file must exist, contain at least one data row, and provide the
    vuelo and tile columns. Raises DatasetError on any violation.
    """
    if not manifest_path.is_file():
        raise DatasetError(f"manifest not found: {manifest_path}")
    try:
        with manifest_path.open(newline="", encoding="utf-8") as fh:
            rows = list(csv.DictReader(fh))
    except OSError as exc:
        raise DatasetError(f"cannot read manifest {manifest_path}: {exc}") from exc

    if not rows:
        raise DatasetError(f"manifest has no data rows: {manifest_path}")
    for column in ("vuelo", "tile"):
        if column not in rows[0]:
            raise DatasetError(
                f"manifest is missing required column {column!r}: {manifest_path}"
            )
    bad_rows = [
        lineno
        for lineno, row in enumerate(rows, start=2)
        if not row.get("vuelo") or not row.get("tile")
    ]
    if bad_rows:
        raise DatasetError(
            f"manifest has empty vuelo/tile values at row(s): {bad_rows}"
        )
    return rows


def write_layout(
    manifest_path: Path,
    labels_root: Path,
    output_dir: Path,
    valid_ratio: float = DEFAULT_VALID_RATIO,
    seed: int = DEFAULT_SEED,
    names: list[str] | None = None,
) -> dict:
    """Write a complete YOLO dataset layout from a manifest + tiles root.

    FR-4: ALL inputs (manifest shape, image existence, sidecar content) are
    validated BEFORE --output is created; on any error nothing is written.
    Reruns overwrite deterministically (NFR-1): same inputs + seed produce a
    byte-identical tree.
    """
    if not 0.0 < valid_ratio < 1.0:
        raise DatasetError(f"valid_ratio must be in (0, 1), got {valid_ratio}")

    rows = read_manifest(manifest_path)
    groups = group_flights(rows)

    # Phase 1 — validate everything before touching the output dir.
    for flight in sorted(groups):
        flight_dir = labels_root / flight
        for row in groups[flight]:
            image = flight_dir / row["tile"]
            if not image.is_file():
                raise DatasetError(f"image not found: {image}")
            sidecar = image.with_suffix(".txt")
            if sidecar.exists():
                validate_label_file(sidecar)

    # Phase 2 — write. mkdir(..., exist_ok=True) + plain copies make reruns
    # overwrite the previous tree instead of failing (FR-4 idempotency).
    for split in ("train", "valid"):
        (output_dir / "images" / split).mkdir(parents=True, exist_ok=True)
        (output_dir / "labels" / split).mkdir(parents=True, exist_ok=True)

    summary = {
        "flights": len(groups),
        "train_images": 0,
        "valid_images": 0,
        "background_images": 0,
    }
    for flight in sorted(groups):
        flight_dir = labels_root / flight
        tiles = [row["tile"] for row in groups[flight]]
        train_tiles, valid_tiles = split_tiles(tiles, valid_ratio=valid_ratio, seed=seed)
        for split, split_tiles_ in (("train", train_tiles), ("valid", valid_tiles)):
            images_dir = output_dir / "images" / split
            labels_dir = output_dir / "labels" / split
            for tile in split_tiles_:
                shutil.copyfile(flight_dir / tile, images_dir / tile)
                summary["train_images" if split == "train" else "valid_images"] += 1
                copied = copy_label((flight_dir / tile).with_suffix(".txt"), labels_dir)
                if copied is None:
                    summary["background_images"] += 1

    (output_dir / "data.yaml").write_text(
        yaml.safe_dump(build_yaml(output_dir, names), sort_keys=False),
        encoding="utf-8",
    )
    return summary


def main(argv: list[str] | None = None) -> int:
    """CLI entry point (FR-4). Returns the process exit code.

    0 = success; 1 = invalid input / dataset error (nothing written);
    2 = usage error (--valid-ratio out of range; argparse itself exits 2 on
    missing required arguments).
    """
    parser = argparse.ArgumentParser(
        prog="prepare_dataset.py",
        description="Build a YOLO dataset from tile_pipeline.py tiles and labels.",
    )
    parser.add_argument("--manifest", required=True, help="path to tile_pipeline.py manifest.csv")
    parser.add_argument(
        "--labels",
        required=True,
        help="tiles root with per-flight image and label folders",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="dataset output directory (overwritten deterministically on rerun)",
    )
    parser.add_argument(
        "--valid-ratio",
        type=float,
        default=DEFAULT_VALID_RATIO,
        help="validation fraction per flight, in (0, 1) (default: 0.2)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help="random seed for reproducible splits (default: 42)",
    )
    args = parser.parse_args(argv)

    if not 0.0 < args.valid_ratio < 1.0:
        print(f"Error: --valid-ratio must be in (0, 1), got {args.valid_ratio}", file=sys.stderr)
        return 2

    try:
        summary = write_layout(
            manifest_path=Path(args.manifest),
            labels_root=Path(args.labels),
            output_dir=Path(args.output),
            valid_ratio=args.valid_ratio,
            seed=args.seed,
        )
    except (DatasetError, LabelFormatError, OSError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    print(
        f"Dataset ready: {summary['train_images']} train / {summary['valid_images']} valid "
        f"images across {summary['flights']} flights "
        f"({summary['background_images']} background, image-only)."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
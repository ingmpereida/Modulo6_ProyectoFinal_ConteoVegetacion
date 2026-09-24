#!/usr/bin/env python3
"""infer.py — batch plant counting from YOLO predictions.

PR1 shipped the importable geometry core: the pixel-space xyxy Box contract
plus the filter/project/clip/IoU/NMS helpers used by the counting pipeline.
PR2 layers the counting pipeline on top: manifest-driven tile specs, PIL
image loading (PNG/JPEG/TIF -> RGB numpy arrays), the per-photo counting
orchestration, and deterministic CSV/summary rendering. PR3 closes the
module: ``load_model`` is the single seam to the ultralytics detector
(lazy-imported inside the function, so module import and ``--help`` never
pull the heavy runtime in — NFR-1), and ``main()`` wires the command-line
contract (FR-1/FR-2: argument parser, exit codes, input modes,
validate-then-write, and the pinned summary/verbose lines). Everything stays
pure and deterministic (NFR-2) and consumes the predictor duck-type
`predict(image) -> [Box]`, so the detector runtime is never needed to import
or test this module.
"""

import argparse
import csv
import io
import itertools
import os
import sys
import tempfile
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image


__all__ = [
    "Box",
    "TileSpec",
    "PhotoResult",
    "Predictor",
    "filter_boxes",
    "project_box",
    "clip_box",
    "box_iou",
    "nms_dedup",
    "load_image",
    "read_tiles",
    "count_photo",
    "render_csv",
    "render_summaries",
    "load_model",
    "main",
]


@dataclass(frozen=True)
class Box:
    """A detection in tile-pixel space (x1, y1, x2, y2), confidence, class.

    Coordinates are `xyxy` in tile pixels — never normalized — so projecting
    a box into photo space is a plain offset translation (design D1).
    """

    xyxy: tuple[float, float, float, float]
    conf: float
    cls: int


def filter_boxes(boxes: list[dict], conf: float, cls: int = 0) -> list[Box]:
    """Keep detections of ``cls`` with confidence ``>= conf``, in input order.

    ``boxes`` is the raw predictor output: dicts with ``xyxy`` (list of 4
    floats), ``conf`` (float), ``cls`` (int). Returns frozen Boxes, so later
    pipeline stages (project/clip/NMS) can rely on immutability (FR-3).
    """
    kept: list[Box] = []
    for raw in boxes:
        if raw["cls"] == cls and raw["conf"] >= conf:
            kept.append(Box(tuple(raw["xyxy"]), raw["conf"], raw["cls"]))
    return kept


def project_box(box: Box, x_off: int, y_off: int) -> Box:
    """Translate a tile-pixel box into photo space by the tile's offset.

    Boxes are plain pixel ``xyxy``, so projection is a pure coordinate
    addition on all four values — no normalization math (FR-3 amended, D1).
    Confidence and class pass through untouched.
    """
    x1, y1, x2, y2 = box.xyxy
    return Box((x1 + x_off, y1 + y_off, x2 + x_off, y2 + y_off), box.conf, box.cls)


def clip_box(box: Box, photo_w: int, photo_h: int) -> Box | None:
    """Clip a projected box to the photo rectangle (0, 0, photo_w, photo_h).

    Partially overlapping boxes are clipped to the bounds and still counted
    (border tiles keep their plants, FR-4); boxes with no positive-area
    overlap are fully outside the photo and discarded (design D5). ``None``
    means the box contributes nothing.
    """
    x1, y1, x2, y2 = box.xyxy
    cx1, cy1 = max(x1, 0.0), max(y1, 0.0)
    cx2, cy2 = min(x2, float(photo_w)), min(y2, float(photo_h))
    if cx2 <= cx1 or cy2 <= cy1:
        return None
    return Box((cx1, cy1, cx2, cy2), box.conf, box.cls)


def box_iou(a: Box, b: Box) -> float:
    """Intersection-over-union of two pixel-space boxes (FR-4).

    Valid for any position (overlapping, touching, disjoint, degenerate):
    zero-area intersections or unions return 0.0 instead of dividing by
    zero, so NMS never breaks on empty boxes.
    """
    ax1, ay1, ax2, ay2 = a.xyxy
    bx1, by1, bx2, by2 = b.xyxy
    inter_w = min(ax2, bx2) - max(ax1, bx1)
    inter_h = min(ay2, by2) - max(ay1, by1)
    if inter_w <= 0.0 or inter_h <= 0.0:
        return 0.0
    inter = inter_w * inter_h
    union = (ax2 - ax1) * (ay2 - ay1) + (bx2 - bx1) * (by2 - by1) - inter
    if union <= 0.0:
        return 0.0
    return inter / union


def nms_dedup(boxes: list[Box], iou: float) -> list[Box]:
    """Greedy global IoU NMS: keep a box iff IoU with every kept box < iou.

    Deterministic by construction (FR-4, NFR-2): boxes are processed sorted
    by ``(-conf, original_index)`` — highest confidence first, ties broken
    by input position — and the result comes back in that order (D3). A box
    whose IoU with any already-kept box is ``>= iou`` is a duplicate of the
    same plant and is suppressed (merge semantics, D2).
    """
    ordered = sorted(enumerate(boxes), key=lambda pair: (-pair[1].conf, pair[0]))
    kept: list[Box] = []
    for _, box in ordered:
        if all(box_iou(box, kept_box) < iou for kept_box in kept):
            kept.append(box)
    return kept


@dataclass(frozen=True)
class TileSpec:
    """One manifest row: a tile image and where it sits inside its photo.

    ``path`` lives under ``tiles_root / flight`` (tile_pipeline.py layout);
    offsets and dims come from the manifest columns verbatim (D9). Photo
    bounds are needed by clip_box (D5) and come from the photo columns.
    """

    path: Path
    flight: str
    photo: str
    x_offset: int
    y_offset: int
    tile_w: int
    tile_h: int
    photo_w: int
    photo_h: int


@dataclass(frozen=True)
class PhotoResult:
    """Count outcome for one photo after per-photo global NMS dedup (FR-4)."""

    global_count: int
    box_count: int
    dedup_removed: int
    source_tiles: tuple[str, ...]


def load_image(path: Path | str) -> np.ndarray:
    """Decode any Pillow-readable image (PNG/JPEG/TIF) to RGB H×W×3 uint8.

    The predictor duck-type consumes exactly this array shape (FR-6), and
    the decode never touches the heavy detector runtime (NFR-1). A missing
    file raises FileNotFoundError; an existing file that cannot be decoded
    as an image raises ValueError.
    """
    img_path = Path(path)
    if not img_path.is_file():
        raise FileNotFoundError(f"image file not found: {img_path}")
    try:
        with Image.open(img_path) as img:
            return np.asarray(img.convert("RGB"), dtype=np.uint8)
    except (ValueError, OSError) as exc:
        raise ValueError(f"cannot decode image: {img_path}") from exc


def read_tiles(manifest_path: Path, tiles_root: Path) -> list[TileSpec]:
    """Parse a tile_pipeline manifest.csv into TileSpecs, in manifest row order.

    Column contract (tile_pipeline.py): vuelo, imagen_origen, tile,
    x_offset, y_offset, tile_w, tile_h, imagen_ancho, imagen_alto. Each tile
    resolves to ``tiles_root / vuelo / tile``; a manifest that references a
    missing tile image raises FileNotFoundError (the CLI maps it to exit 1).
    Tile paths are CONFINED to the tiles root (R1-001): every tile is
    resolved and must stay under ``tiles_root`` — a crafted ``..`` or
    absolute component that would escape the root raises FileNotFoundError
    naming the offending row/tile.
    """
    tiles: list[TileSpec] = []
    root_resolved = Path(tiles_root).resolve()
    with Path(manifest_path).open("r", newline="", encoding="utf-8") as fh:
        for row_no, row in enumerate(csv.DictReader(fh), start=2):  # header is line 1
            tile_path = root_resolved / row["vuelo"] / row["tile"]
            tile_resolved = tile_path.resolve()
            if not tile_resolved.is_relative_to(root_resolved):
                raise FileNotFoundError(
                    f"manifest row {row_no} tile {row['tile']!r} escapes the "
                    f"tiles root: {tile_resolved} (expected under {root_resolved})"
                )
            if not tile_resolved.is_file():
                raise FileNotFoundError(
                    f"manifest references missing tile image: {tile_resolved}"
                )
            tiles.append(
                TileSpec(
                    path=tile_resolved,
                    flight=row["vuelo"],
                    photo=row["imagen_origen"],
                    x_offset=int(row["x_offset"]),
                    y_offset=int(row["y_offset"]),
                    tile_w=int(row["tile_w"]),
                    tile_h=int(row["tile_h"]),
                    photo_w=int(row["imagen_ancho"]),
                    photo_h=int(row["imagen_alto"]),
                )
            )
    return tiles


def count_photo(
    tiles: list[TileSpec], load_image, predict, conf: float, iou: float
) -> PhotoResult:
    """Count the plants in one photo from its tiles, in manifest row order.

    Per tile: load -> predict -> filter (class + conf) -> project to photo
    space -> clip to photo bounds (FR-3, D5); ``box_count`` accumulates the
    filter-passing detections. Per photo: global IoU NMS dedup (FR-4, D2)
    yields ``global_count``; ``dedup_removed = box_count - global_count``
    (FR-4) and ``source_tiles`` = sorted names of the tiles that contributed
    any box to the counting pool — both tiles of a merged duplicate are
    listed (FR-4 scenario). A photo with a single tile skips dedup: with no
    second offset there is nothing to dedup against (D6 / FR-2 scenario).
    """
    if not tiles:
        raise ValueError("count_photo requires at least one tile")
    photo_w, photo_h = tiles[0].photo_w, tiles[0].photo_h
    pool: list[tuple[str, Box]] = []
    box_count = 0
    for tile in tiles:  # manifest row order, never re-sorted (NFR-3)
        for box in filter_boxes(predict(load_image(tile.path)), conf):
            box_count += 1
            clipped = clip_box(
                project_box(box, tile.x_offset, tile.y_offset), photo_w, photo_h
            )
            if clipped is not None:
                pool.append((tile.path.name, clipped))
    source_tiles = tuple(sorted({name for name, _ in pool}))
    if len(tiles) <= 1:
        global_count = len(pool)
    else:
        global_count = len(nms_dedup([box for _, box in pool], iou))
    return PhotoResult(
        global_count=global_count,
        box_count=box_count,
        dedup_removed=box_count - global_count,
        source_tiles=source_tiles,
    )


CSV_FIELDS = [
    "flight",
    "photo",
    "global_count",
    "box_count",
    "dedup_removed",
    "source_tiles",
]


def render_csv(rows: list[dict]) -> str:
    """Render counting rows to deterministic CSV text (FR-5, NFR-3).

    Exact header (flight,photo,global_count,box_count,dedup_removed,
    source_tiles), rows sorted by (flight, photo), ``source_tiles`` joined
    with ';'. Pure string output — the caller decides when to write it (D7).
    """
    buf = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=CSV_FIELDS, lineterminator="\n")
    writer.writeheader()
    for row in sorted(rows, key=lambda r: (r["flight"], r["photo"])):
        out = dict(row)
        out["source_tiles"] = ";".join(row["source_tiles"])
        writer.writerow(out)
    return buf.getvalue()


def render_summaries(rows: list[dict]) -> list[str]:
    """Per-flight summary lines in a PINNED format (finding 6; PR3 asserts):
    ``summary flight={flight} photos={n} plants={sum(global_count)}`` over
    rows sorted by (flight, photo), flights in first-seen order.
    """
    summaries: list[str] = []
    ordered = sorted(rows, key=lambda r: (r["flight"], r["photo"]))
    for flight, group in itertools.groupby(ordered, key=lambda r: r["flight"]):
        group_rows = list(group)
        summaries.append(
            f"summary flight={flight} photos={len(group_rows)} "
            f"plants={sum(r['global_count'] for r in group_rows)}"
        )
    return summaries


# ---------------------------------------------------------------------------
# PR3: predictor seam + command-line entry point
# ---------------------------------------------------------------------------


Predictor = Callable[[np.ndarray], list[dict]]  # .predict(img) -> [{xyxy, conf, cls}]


class _UltralyticsPredictor:
    """Adapt an ultralytics YOLO to the Predictor duck-type (pixel-xyxy dicts).

    Maps ``results[0].boxes.xyxy/conf/cls`` (tensors) to the pixel-space dict
    contract ``{xyxy: [4 floats], conf: float, cls: int}`` (design D1), so the
    rest of the pipeline sees the same shape FakeModel produces. Fails fast
    (RuntimeError) when the result shape drifts from what the pipeline
    expects, so a silent counting corruption can never pass unnoticed.
    """

    def __init__(self, model):
        self._model = model

    def predict(self, image: np.ndarray) -> list[dict]:
        results = self._model.predict(image)
        try:
            boxes = results[0].boxes
            if boxes is None:
                # Zero detections (ultralytics 8.4.x sets boxes=None for an
                # empty photo): empty photos are the norm in plant counting,
                # so yield no detections instead of aborting the batch
                # (R4-001 — overrides the previous fail-fast on no boxes).
                return []
            xyxy_rows = boxes.xyxy.tolist()
            confs = boxes.conf.tolist()
            clss = boxes.cls.tolist()
        except (AttributeError, IndexError, TypeError) as exc:
            raise RuntimeError(
                "unexpected ultralytics result shape: expected results[0].boxes "
                "with xyxy/conf/cls tensors (pixel-xyxy contract, design D1)"
            ) from exc
        out: list[dict] = []
        for i, row in enumerate(xyxy_rows):
            out.append(
                {
                    "xyxy": [float(v) for v in row],
                    "conf": float(_scalar(confs[i])),
                    "cls": int(_scalar(clss[i])),
                }
            )
        return out


def _scalar(value):
    """Unwrap a one-level-nested scalar (ultralytics emits (N,1) tensors sometimes)."""
    if isinstance(value, (list, tuple)) and len(value) == 1:
        return value[0]
    return value


def load_model(weights: str) -> Predictor:
    """Build the predictor for ``weights`` — the module's ONLY ultralytics seam.

    The YOLO runtime is imported lazily INSIDE this function, so importing
    infer or running ``--help`` never pulls ultralytics in (NFR-1). The CLI
    rejects a missing/unreadable path earlier (FR-1, exit 2); here, any load
    failure — e.g. corrupt-but-readable weights — raises and the CLI maps it
    to exit 1 (D8).
    """
    from ultralytics import YOLO

    return _UltralyticsPredictor(YOLO(weights))


def _existing_file(value: str) -> Path:
    """argparse type: the path must be an existing regular file (finding 4).

    ``Path.is_file()`` rejects both missing paths and directories, which is
    the exact FR-1 "missing/unreadable --weights -> exit 2" gate. ``os.access``
    is deliberately avoided (unreliable ACLs on win32).
    """
    path = Path(value)
    if not path.is_file():
        raise argparse.ArgumentTypeError(f"weights file not found or not readable: {value}")
    return path


def _bounded_float(flag: str) -> Callable[[str], float]:
    """argparse type factory for an open (0, 1) interval (D4)."""

    def _parse(value: str) -> float:
        try:
            number = float(value)
        except ValueError:
            raise argparse.ArgumentTypeError(f"--{flag} must be strictly between 0 and 1") from None
        if not 0.0 < number < 1.0:
            raise argparse.ArgumentTypeError(
                f"--{flag} must be strictly between 0 and 1, got {value}"
            )
        return number

    return _parse


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="infer.py",
        description=(
            "Count plants in drone photos from YOLO detections over "
            "tile_pipeline.py tiles (FR-1)."
        ),
    )
    parser.add_argument(
        "--weights",
        required=True,
        type=_existing_file,
        metavar="PATH",
        help="trained YOLO weights (.pt); must exist and be readable (else exit 2)",
    )
    parser.add_argument(
        "--input",
        required=True,
        metavar="PATH",
        help="single photo (JPEG/PNG/TIF) OR directory containing manifest.csv + tiles/",
    )
    parser.add_argument(
        "--output",
        required=True,
        metavar="CSV",
        help="CSV path; written only after ALL photos succeed (NFR-6)",
    )
    parser.add_argument(
        "--conf",
        default=0.25,
        type=_bounded_float("conf"),
        metavar="F",
        help="minimum detection confidence, strictly between 0 and 1 (default 0.25)",
    )
    parser.add_argument(
        "--iou",
        default=0.5,
        type=_bounded_float("iou"),
        metavar="F",
        help="global NMS IoU threshold, strictly between 0 and 1 (default 0.5)",
    )
    parser.add_argument(
        "--summary", action="store_true", help="print per-flight totals to stdout"
    )
    parser.add_argument(
        "--verbose", action="store_true", help="print per-photo progress lines"
    )
    return parser


def _single_photo_tiles(path: Path) -> list[TileSpec]:
    """Turn one photo into a single full-size tile spec (D6 / FR-2).

    flight = photo = the file stem, offsets (0, 0), dims taken from the
    decoded image. With a single tile the pipeline skips dedup (D6) and the
    CSV row gets ``source_tiles`` = the photo filename.
    """
    image = load_image(path)
    h, w = image.shape[:2]
    return [
        TileSpec(
            path=path,
            flight=path.stem,
            photo=path.stem,
            x_offset=0,
            y_offset=0,
            tile_w=w,
            tile_h=h,
            photo_w=w,
            photo_h=h,
        )
    ]


def _write_output_atomic(output: Path, data: bytes) -> None:
    """Write ``data`` to ``output`` atomically (R1-004/R4-003).

    The bytes go to a temp file in the SAME directory as ``output`` (same
    filesystem, so the final swap is atomic), are flushed and fsync'd, then
    ``os.replace`` moves them over the target. A crash, kill, or disk-full
    mid-write leaves any pre-existing output byte-untouched, and a reader
    never observes a half-written CSV. On failure the temp file is unlinked.
    """
    tmp_path: str | None = None
    try:
        fd, tmp_path = tempfile.mkstemp(
            dir=output.parent, prefix=f"{output.name}.", suffix=".tmp"
        )
        with os.fdopen(fd, "wb") as fh:
            fh.write(data)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp_path, output)
    finally:
        if tmp_path is not None:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass  # already replaced (success) or never created — best effort


def main(argv: list[str] | None = None) -> int:
    """Run the counting CLI and return the process exit code (FR-1).

    0 = success (CSV written); 1 = data/processing error after argument
    validation (nothing written, NFR-6/D7); 2 = invalid usage — argparse
    exits with SystemExit(2) before any work: missing/unreadable weights,
    out-of-range --conf/--iou, missing required args, nonexistent --input.
    """
    args = _build_parser().parse_args(argv)

    input_path = Path(args.input)
    if not input_path.exists():
        print(f"error: input path does not exist: {input_path}", file=sys.stderr)
        return 2

    try:
        if input_path.is_dir():
            manifest = input_path / "manifest.csv"
            if not manifest.is_file():
                print(
                    f"error: missing manifest.csv in input directory: {input_path}",
                    file=sys.stderr,
                )
                return 1
            tiles = read_tiles(manifest, input_path / "tiles")
        else:  # single photo: no manifest, no dedup (D6)
            tiles = _single_photo_tiles(input_path)

        model = load_model(str(args.weights))

        # (flight, photo) groups in first-seen manifest order (NFR-3).
        groups: dict[tuple[str, str], list[TileSpec]] = {}
        for tile in tiles:
            groups.setdefault((tile.flight, tile.photo), []).append(tile)
        if not groups:
            # Empty manifest: not an error (R4-002) — header-only CSV is
            # still written — but the silent success must be observable.
            print(
                f"warning: no photos in manifest {manifest}: header-only CSV "
                "written (no rows produced)",
                file=sys.stderr,
            )

        rows: list[dict] = []
        for (flight, photo), photo_tiles in groups.items():
            result = count_photo(
                photo_tiles, load_image, model.predict, args.conf, args.iou
            )
            if args.verbose:
                print(
                    f"verbose flight={flight} photo={photo} tiles={len(photo_tiles)} "
                    f"boxes={result.box_count} kept={result.global_count}"
                )
            rows.append(
                {
                    "flight": flight,
                    "photo": photo,
                    "global_count": result.global_count,
                    "box_count": result.box_count,
                    "dedup_removed": result.dedup_removed,
                    "source_tiles": result.source_tiles,
                }
            )

        # Validate-then-write (D7): render once, write once atomically (R4-003).
        csv_text = render_csv(rows)
        _write_output_atomic(Path(args.output), csv_text.encode("utf-8"))
        if args.summary:
            for line in render_summaries(rows):
                print(line)
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
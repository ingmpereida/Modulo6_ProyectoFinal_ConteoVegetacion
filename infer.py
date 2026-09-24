#!/usr/bin/env python3
"""infer.py — batch plant counting from YOLO predictions (PR1: pure core).

This PR ships the importable geometry core only: the pixel-space xyxy Box
contract plus the filter/project/clip/IoU/NMS helpers used by the counting
pipeline. Everything here is pure and deterministic (NFR-2) and consumes the
predictor duck-type `predict(image) -> [Box]`, so no YOLO runtime is needed
to import or test this module (NFR-1). The ultralytics seam, photo pipeline,
and argparse CLI arrive in PR2/PR3.
"""

from dataclasses import dataclass


__all__ = [
    "Box",
    "filter_boxes",
    "project_box",
    "clip_box",
    "box_iou",
    "nms_dedup",
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
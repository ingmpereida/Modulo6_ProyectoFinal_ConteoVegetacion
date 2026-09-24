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


__all__ = ["Box", "filter_boxes"]


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
"""Tests for infer.py pure geometry core (PR1: tasks 1.1-1.6).

Covers the pixel-space xyxy Box contract, class+conf filtering, offset
projection, photo-bound clipping, IoU, and deterministic global NMS dedup.
CPU-only and ultralytics-free by design (NFR-1, NFR-2): in this PR the
module is a plain importable library — no CLI, no YOLO runtime.
"""

import dataclasses
import sys
from pathlib import Path

import pytest

import infer


def test_importing_infer_never_imports_ultralytics():
    """NFR-1: importing the module must not pull the YOLO runtime in."""
    assert "ultralytics" not in sys.modules
    assert hasattr(infer, "Box")


# ---------------------------------------------------------------------------
# PR1 1.1: Box dataclass + filter_boxes (class + confidence filter)
# ---------------------------------------------------------------------------


class TestBox:
    def test_carries_pixel_xyxy_conf_and_cls(self):
        box = infer.Box((10.0, 20.0, 30.0, 40.0), 0.9, 0)
        assert box.xyxy == (10.0, 20.0, 30.0, 40.0)
        assert box.conf == 0.9
        assert box.cls == 0

    def test_is_frozen_and_rejects_mutation(self):
        box = infer.Box((1.0, 2.0, 3.0, 4.0), 0.5, 0)
        with pytest.raises(dataclasses.FrozenInstanceError):
            box.conf = 0.99
        assert box.conf == 0.5  # unchanged


class TestFilterBoxes:
    def test_keeps_high_conf_boxes_of_target_class(self):
        boxes = [
            {"xyxy": [10.0, 20.0, 30.0, 40.0], "conf": 0.8, "cls": 0},
            {"xyxy": [50.0, 60.0, 70.0, 80.0], "conf": 0.1, "cls": 0},
        ]
        kept = infer.filter_boxes(boxes, 0.25)

        assert len(kept) == 1
        assert kept[0].xyxy == (10.0, 20.0, 30.0, 40.0)
        assert kept[0].conf == 0.8
        assert kept[0].cls == 0

    def test_conf_equal_to_threshold_is_kept(self):
        boxes = [{"xyxy": [0.0, 0.0, 1.0, 1.0], "conf": 0.25, "cls": 0}]
        kept = infer.filter_boxes(boxes, 0.25)
        assert [b.conf for b in kept] == [0.25]

    def test_other_classes_are_dropped_with_default_filter(self):
        boxes = [
            {"xyxy": [1.0, 2.0, 3.0, 4.0], "conf": 0.9, "cls": 0},
            {"xyxy": [5.0, 6.0, 7.0, 8.0], "conf": 0.9, "cls": 1},
        ]
        kept = infer.filter_boxes(boxes, 0.25)
        assert [b.cls for b in kept] == [0]

    def test_custom_cls_selects_that_class(self):
        boxes = [
            {"xyxy": [1.0, 2.0, 3.0, 4.0], "conf": 0.9, "cls": 0},
            {"xyxy": [5.0, 6.0, 7.0, 8.0], "conf": 0.9, "cls": 2},
        ]
        kept = infer.filter_boxes(boxes, 0.25, cls=2)
        assert [b.cls for b in kept] == [2]

    def test_input_order_is_preserved(self):
        boxes = [
            {"xyxy": [0.0, 0.0, 1.0, 1.0], "conf": 0.4, "cls": 0},
            {"xyxy": [2.0, 2.0, 3.0, 3.0], "conf": 0.9, "cls": 0},
            {"xyxy": [4.0, 4.0, 5.0, 5.0], "conf": 0.7, "cls": 0},
        ]
        kept = infer.filter_boxes(boxes, 0.25)
        assert [b.xyxy for b in kept] == [
            (0.0, 0.0, 1.0, 1.0),
            (2.0, 2.0, 3.0, 3.0),
            (4.0, 4.0, 5.0, 5.0),
        ]

    def test_no_detections_yields_empty_result(self):
        kept = infer.filter_boxes([], 0.25)
        assert kept == []


# ---------------------------------------------------------------------------
# PR1 1.2: project_box — pure offset translation into photo space
# ---------------------------------------------------------------------------


class TestProjectBox:
    def test_x_offset_shifts_all_coordinates(self):
        box = infer.Box((0.0, 0.0, 320.0, 320.0), 0.8, 0)
        projected = infer.project_box(box, 544, 0)
        assert projected.xyxy == (544.0, 0.0, 864.0, 320.0)

    def test_x_and_y_offsets_and_negative_shifts(self):
        box = infer.Box((100.0, 200.0, 300.0, 400.0), 0.6, 0)
        projected = infer.project_box(box, -50, -25)
        assert projected.xyxy == (50.0, 175.0, 250.0, 375.0)

    def test_metadata_passes_through_untouched(self):
        box = infer.Box((1.0, 2.0, 3.0, 4.0), 0.95, 0)
        projected = infer.project_box(box, 1000, 500)
        assert projected.conf == 0.95
        assert projected.cls == 0


# ---------------------------------------------------------------------------
# PR1 1.3: clip_box — clip partial boxes to photo bounds, discard outside
# ---------------------------------------------------------------------------


class TestClipBox:
    def test_inside_box_is_untouched(self):
        box = infer.Box((100.0, 50.0, 300.0, 250.0), 0.7, 0)
        clipped = infer.clip_box(box, 4000, 3000)
        assert clipped == box

    def test_border_partial_box_is_clipped_to_photo_bounds(self):
        box = infer.Box((-100.0, 0.0, 100.0, 320.0), 0.7, 0)
        clipped = infer.clip_box(box, 4000, 3000)
        assert clipped is not None
        assert clipped.xyxy == (0.0, 0.0, 100.0, 320.0)
        assert clipped.conf == 0.7
        assert clipped.cls == 0

    def test_box_sticking_out_bottom_is_clipped(self):
        box = infer.Box((0.0, 2900.0, 200.0, 3100.0), 0.7, 0)
        clipped = infer.clip_box(box, 4000, 3000)
        assert clipped.xyxy == (0.0, 2900.0, 200.0, 3000.0)

    def test_fully_outside_right_is_discarded(self):
        box = infer.Box((4100.0, 0.0, 4200.0, 100.0), 0.7, 0)
        assert infer.clip_box(box, 4000, 3000) is None

    def test_fully_outside_left_is_discarded(self):
        box = infer.Box((-200.0, 0.0, -50.0, 100.0), 0.7, 0)
        assert infer.clip_box(box, 4000, 3000) is None


# ---------------------------------------------------------------------------
# PR1 1.4: box_iou — intersection over union for two boxes
# ---------------------------------------------------------------------------


class TestBoxIou:
    def test_identical_box_has_iou_1(self):
        box = infer.Box((0.0, 0.0, 10.0, 10.0), 0.9, 0)
        assert infer.box_iou(box, box) == 1.0

    def test_disjoint_boxes_have_iou_0(self):
        a = infer.Box((0.0, 0.0, 10.0, 10.0), 0.9, 0)
        b = infer.Box((20.0, 20.0, 30.0, 30.0), 0.9, 0)
        assert infer.box_iou(a, b) == 0.0

    def test_half_width_overlap_is_exactly_0_5(self):
        a = infer.Box((0.0, 0.0, 6.0, 10.0), 0.9, 0)
        b = infer.Box((2.0, 0.0, 8.0, 10.0), 0.9, 0)
        # inter = 4*10 = 40, union = 60 + 60 - 40 = 80
        assert infer.box_iou(a, b) == 0.5

    def test_partial_2d_overlap(self):
        a = infer.Box((0.0, 0.0, 4.0, 4.0), 0.9, 0)
        b = infer.Box((2.0, 2.0, 6.0, 6.0), 0.9, 0)
        # inter = 2*2 = 4, union = 16 + 16 - 4 = 28
        assert infer.box_iou(a, b) == pytest.approx(4 / 28)

    def test_contained_box(self):
        a = infer.Box((0.0, 0.0, 10.0, 10.0), 0.9, 0)
        b = infer.Box((2.0, 2.0, 8.0, 8.0), 0.9, 0)
        assert infer.box_iou(a, b) == pytest.approx(36 / 100)

    def test_zero_area_box_returns_0_without_division_by_zero(self):
        a = infer.Box((0.0, 0.0, 0.0, 10.0), 0.9, 0)  # zero width — degenerate
        b = infer.Box((0.0, 0.0, 10.0, 10.0), 0.9, 0)
        assert infer.box_iou(a, b) == 0.0


# ---------------------------------------------------------------------------
# PR1 1.5: nms_dedup — deterministic global IoU NMS (FR-4, design D2/D3)
# ---------------------------------------------------------------------------


class TestNmsDedup:
    def test_duplicate_boxes_merge_keeping_highest_conf(self):
        a = infer.Box((0.0, 0.0, 320.0, 320.0), 0.9, 0)
        b = infer.Box((0.0, 0.0, 320.0, 320.0), 0.8, 0)
        kept = infer.nms_dedup([a, b], 0.5)
        assert [k.conf for k in kept] == [0.9]

    def test_disjoint_boxes_all_survive_in_conf_order(self):
        a = infer.Box((0.0, 0.0, 100.0, 100.0), 0.9, 0)
        b = infer.Box((200.0, 0.0, 300.0, 100.0), 0.8, 0)
        kept = infer.nms_dedup([b, a], 0.5)  # input not sorted — output must be
        assert [k.conf for k in kept] == [0.9, 0.8]

    def test_low_overlap_below_threshold_survives(self):
        a = infer.Box((0.0, 0.0, 100.0, 100.0), 0.9, 0)
        b = infer.Box((70.0, 0.0, 170.0, 100.0), 0.8, 0)  # IoU = 30/170 ~ 0.18
        kept = infer.nms_dedup([a, b], 0.5)
        assert len(kept) == 2

    def test_exact_threshold_overlap_merges(self):
        # IoU exactly 0.5 == --iou -> suppressed (>= threshold merges, D2/D4)
        a = infer.Box((0.0, 0.0, 6.0, 10.0), 0.9, 0)
        b = infer.Box((2.0, 0.0, 8.0, 10.0), 0.9, 0)
        kept = infer.nms_dedup([a, b], 0.5)
        assert [k.xyxy for k in kept] == [(0.0, 0.0, 6.0, 10.0)]

    def test_equal_conf_tie_breaks_by_lower_original_index(self):
        # Same conf: the FIRST box in the input wins the merge (D3) — no
        # salted hash() or RNG anywhere near the ordering.
        first = infer.Box((0.0, 0.0, 10.0, 10.0), 0.8, 0)
        second = infer.Box((1.0, 0.0, 11.0, 10.0), 0.8, 0)  # IoU ~ 0.82
        kept = infer.nms_dedup([first, second], 0.5)
        assert kept == [first]

    def test_high_conf_suppresses_later_duplicates_but_keeps_distinct(self):
        a = infer.Box((0.0, 0.0, 100.0, 100.0), 0.9, 0)
        b = infer.Box((2.0, 2.0, 102.0, 102.0), 0.6, 0)  # IoU with a ~ 0.92
        c = infer.Box((4.0, 4.0, 104.0, 104.0), 0.5, 0)  # IoU with a ~ 0.85
        d = infer.Box((300.0, 0.0, 400.0, 100.0), 0.85, 0)  # disjoint
        kept = infer.nms_dedup([a, b, c, d], 0.5)
        assert [k.conf for k in kept] == [0.9, 0.85]

    def test_kept_order_is_confidence_descending(self):
        boxes = [
            infer.Box((0.0, 0.0, 100.0, 100.0), 0.7, 0),
            infer.Box((200.0, 0.0, 300.0, 100.0), 0.9, 0),
            infer.Box((400.0, 0.0, 500.0, 100.0), 0.8, 0),
        ]
        kept = infer.nms_dedup(boxes, 0.5)
        assert [k.conf for k in kept] == [0.9, 0.8, 0.7]
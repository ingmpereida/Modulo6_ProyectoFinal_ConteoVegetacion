"""Tests for infer.py pure geometry core (PR1: tasks 1.1-1.6).

Covers the pixel-space xyxy Box contract, class+conf filtering, offset
projection, photo-bound clipping, IoU, and deterministic global NMS dedup.
CPU-only and ultralytics-free by design (NFR-1, NFR-2): in this PR the
module is a plain importable library — no CLI, no YOLO runtime.
"""

import dataclasses
import io
import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

import infer
from conftest import FakeModel


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


# ---------------------------------------------------------------------------
# PR1 1.6: gate — importable library only, ultralytics-free (NFR-1, NFR-2)
# ---------------------------------------------------------------------------


class TestPr1Gate:
    def test_ultralytics_is_seam_only_and_never_imported_at_module_level(self):
        # PR3 moved the PR1 boundary: the detector runtime is now NAMED in
        # infer.py, but only inside load_model (lazy import, NFR-1) — a
        # module-level import would break --help and the unit suite.
        src = Path(infer.__file__).read_text(encoding="utf-8")
        lazy_imports = [
            line
            for line in src.splitlines()
            if line.lstrip().startswith(("import ultralytics", "from ultralytics"))
        ]
        assert len(lazy_imports) == 1
        assert lazy_imports[0].startswith("    ")  # indented: inside load_model()

    def test_source_is_importable_library_without_cli(self):
        src = Path(infer.__file__).read_text(encoding="utf-8")
        assert "argparse" not in src
        assert "__main__" not in src
        assert "sys.exit" not in src


# ---------------------------------------------------------------------------
# PR2 2.1: TileSpec + PhotoResult types
# ---------------------------------------------------------------------------


class TestTileSpec:
    def test_carries_path_flight_photo_and_layout_fields(self):
        spec = infer.TileSpec(
            path=Path("tiles/F1/t1.png"),
            flight="F1",
            photo="DJI_0001.JPG",
            x_offset=320,
            y_offset=0,
            tile_w=640,
            tile_h=640,
            photo_w=1280,
            photo_h=800,
        )
        assert spec.path == Path("tiles/F1/t1.png")
        assert spec.flight == "F1"
        assert spec.photo == "DJI_0001.JPG"
        assert (spec.x_offset, spec.y_offset) == (320, 0)
        assert (spec.tile_w, spec.tile_h) == (640, 640)
        assert (spec.photo_w, spec.photo_h) == (1280, 800)

    def test_is_frozen_and_rejects_mutation(self):
        spec = infer.TileSpec(Path("t.png"), "F", "p.jpg", 0, 0, 640, 640, 4000, 3000)
        with pytest.raises(dataclasses.FrozenInstanceError):
            spec.x_offset = 999
        assert spec.x_offset == 0


class TestPhotoResult:
    def test_carries_count_fields_and_source_tiles_tuple(self):
        result = infer.PhotoResult(
            global_count=1, box_count=2, dedup_removed=1,
            source_tiles=("t1.png", "t2.png"),
        )
        assert result.global_count == 1
        assert result.box_count == 2
        assert result.dedup_removed == 1
        assert result.source_tiles == ("t1.png", "t2.png")

    def test_is_frozen_and_rejects_mutation(self):
        result = infer.PhotoResult(3, 4, 1, ("a.png",))
        with pytest.raises(dataclasses.FrozenInstanceError):
            result.global_count = 99
        assert result.global_count == 3


# ---------------------------------------------------------------------------
# PR2 2.2: load_image — PIL decode to RGB uint8 numpy array (no ultralytics)
# ---------------------------------------------------------------------------


def _image_bytes(fmt, mode, size, color):
    """Render a tiny real image with Pillow so load_image has a decodable file."""
    buf = io.BytesIO()
    Image.new(mode, size, color).save(buf, format=fmt)
    return buf.getvalue()


class TestLoadImage:
    def test_loads_png_as_rgb_uint8_array(self, tmp_path):
        path = tmp_path / "tile.png"
        path.write_bytes(_image_bytes("PNG", "RGB", (2, 3), (7, 8, 9)))
        arr = infer.load_image(path)
        assert isinstance(arr, np.ndarray)
        assert arr.shape == (3, 2, 3)  # H, W, 3
        assert arr.dtype == np.uint8
        assert arr[0, 0].tolist() == [7, 8, 9]
        assert arr[2, 1].tolist() == [7, 8, 9]

    def test_loads_tif_as_rgb_uint8_array(self, tmp_path):
        path = tmp_path / "tile.tif"
        path.write_bytes(_image_bytes("TIFF", "RGB", (2, 2), (200, 100, 50)))
        arr = infer.load_image(path)
        assert arr.shape == (2, 2, 3)
        assert arr.dtype == np.uint8
        assert arr[0, 0].tolist() == [200, 100, 50]

    def test_loads_jpeg_as_rgb_uint8_array(self, tmp_path):
        path = tmp_path / "tile.jpg"
        path.write_bytes(_image_bytes("JPEG", "RGB", (5, 4), (10, 20, 30)))
        arr = infer.load_image(path)
        assert isinstance(arr, np.ndarray)
        assert arr.shape == (4, 5, 3)  # JPEG is lossy — assert shape/dtype only
        assert arr.dtype == np.uint8

    def test_grayscale_and_rgba_are_converted_to_three_channels(self, tmp_path):
        gray = tmp_path / "gray.png"
        gray.write_bytes(_image_bytes("PNG", "L", (3, 2), 128))
        arr = infer.load_image(gray)
        assert arr.shape == (2, 3, 3)
        assert arr[0, 0].tolist() == [128, 128, 128]
        rgba = tmp_path / "rgba.png"
        rgba.write_bytes(_image_bytes("PNG", "RGBA", (2, 2), (10, 20, 30, 255)))
        arr2 = infer.load_image(rgba)
        assert arr2.shape == (2, 2, 3)
        assert arr2[0, 0].tolist() == [10, 20, 30]

    def test_non_image_file_raises_value_error(self, tmp_path):
        path = tmp_path / "not_an_image.png"
        path.write_bytes(b"definitely not image bytes")
        with pytest.raises(ValueError):
            infer.load_image(path)

    def test_missing_file_raises_file_not_found_error(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            infer.load_image(tmp_path / "missing.png")


# ---------------------------------------------------------------------------
# PR2 2.4: FakeModel predictor + make_tile_input real-image layout (conftest)
# ---------------------------------------------------------------------------


class TestFakeModel:
    def test_predict_returns_canned_boxes_keyed_on_first_pixel_byte(self):
        model = FakeModel({5: [{"xyxy": [0, 0, 10, 10], "conf": 0.9, "cls": 0}]})
        out = model.predict(np.full((4, 4, 3), 5, dtype=np.uint8))
        assert out == [{"xyxy": [0, 0, 10, 10], "conf": 0.9, "cls": 0}]

    def test_unknown_marker_yields_no_detections(self):
        model = FakeModel({5: [{"xyxy": [0, 0, 10, 10], "conf": 0.9, "cls": 0}]})
        assert model.predict(np.full((2, 2, 3), 9, dtype=np.uint8)) == []


class TestMakeTileInput:
    def test_builds_real_decodable_tiles_and_manifest(self, make_tile_input):
        manifest, tiles_root = make_tile_input(
            [
                {
                    "flight": "F1",
                    "photo": "DJI_0001.JPG",
                    "photo_w": 1280,
                    "photo_h": 800,
                    "tiles": [("t1.png", 0, 0, 640, 640, 1), ("t4.png", 0, 640, 640, 160, 4)],
                }
            ]
        )
        assert manifest.is_file()
        assert (tiles_root / "F1" / "t1.png").is_file()
        assert infer.load_image(tiles_root / "F1" / "t1.png").shape == (640, 640, 3)

    def test_tiles_carry_distinct_stable_markers(self, make_tile_input):
        manifest, tiles_root = make_tile_input(
            [
                {
                    "flight": "F1",
                    "photo": "DJI_0001.JPG",
                    "photo_w": 1280,
                    "photo_h": 800,
                    "tiles": [("t1.png", 0, 0, 640, 640, 1), ("t2.png", 320, 0, 640, 640, 2)],
                }
            ]
        )
        markers = [
            int(infer.load_image(tiles_root / "F1" / name)[0, 0, 0])
            for name in ("t1.png", "t2.png")
        ]
        assert markers == [1, 2]


# ---------------------------------------------------------------------------
# PR2 2.3: read_tiles — manifest.csv -> manifest-row-ordered TileSpec list
# ---------------------------------------------------------------------------


class TestReadTiles:
    def test_parses_fields_paths_and_preserves_row_order(self, make_tile_input):
        manifest, root = make_tile_input(
            [
                {
                    "flight": "F1",
                    "photo": "DJI_0001.JPG",
                    "photo_w": 1280,
                    "photo_h": 800,
                    "tiles": [("t1.png", 0, 0, 640, 640, 1), ("t2.png", 320, 0, 640, 640, 2)],
                }
            ]
        )
        tiles = infer.read_tiles(manifest, root)
        assert [t.path.name for t in tiles] == ["t1.png", "t2.png"]  # row order kept
        first = tiles[0]
        assert first.flight == "F1"
        assert first.photo == "DJI_0001.JPG"
        assert (first.x_offset, first.y_offset) == (0, 0)
        assert (first.tile_w, first.tile_h) == (640, 640)
        assert (first.photo_w, first.photo_h) == (1280, 800)
        assert first.path == root / "F1" / "t1.png"

    def test_missing_referenced_tile_raises_file_not_found(self, make_tile_input):
        manifest, root = make_tile_input(
            [
                {
                    "flight": "F1",
                    "photo": "DJI_0001.JPG",
                    "photo_w": 1280,
                    "photo_h": 800,
                    "tiles": [("t1.png", 0, 0, 640, 640, 1)],
                }
            ]
        )
        (root / "F1" / "t1.png").unlink()
        with pytest.raises(FileNotFoundError):
            infer.read_tiles(manifest, root)

    def test_missing_manifest_file_raises_file_not_found(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            infer.read_tiles(tmp_path / "manifest.csv", tmp_path / "tiles")

    def test_empty_manifest_yields_no_tiles(self, tmp_path):
        manifest = tmp_path / "manifest.csv"
        manifest.write_text(
            "vuelo,imagen_origen,tile,x_offset,y_offset,tile_w,tile_h,imagen_ancho,imagen_alto\n",
            encoding="utf-8",
        )
        assert infer.read_tiles(manifest, tmp_path / "tiles") == []


# ---------------------------------------------------------------------------
# PR2 2.5: count_photo — per-photo pipeline (load->predict->filter->project->
# clip->global NMS), via FakeModel integration (hand-computed expectations)
# ---------------------------------------------------------------------------


class TestCountPhoto:
    def test_duplicate_across_tiles_counts_once(self, make_tile_input):
        manifest, root = make_tile_input(
            [
                {
                    "flight": "F2",
                    "photo": "DJI_0002.JPG",
                    "photo_w": 960,
                    "photo_h": 640,
                    "tiles": [("t1.png", 0, 0, 640, 640, 11), ("t2.png", 320, 0, 640, 640, 12)],
                }
            ]
        )
        tiles = infer.read_tiles(manifest, root)
        model = FakeModel(
            {
                11: [{"xyxy": [400.0, 200.0, 480.0, 280.0], "conf": 0.9, "cls": 0}],
                12: [{"xyxy": [80.0, 200.0, 160.0, 280.0], "conf": 0.8, "cls": 0}],
            }
        )
        result = infer.count_photo(tiles, infer.load_image, model.predict, 0.25, 0.5)
        assert result.global_count == 1  # same plant seen from both tiles
        assert result.box_count == 2
        assert result.dedup_removed == 1
        assert result.source_tiles == ("t1.png", "t2.png")  # both contributors listed

    def test_border_partial_and_distinct_plants_survive(self, make_tile_input):
        manifest, root = make_tile_input(
            [
                {
                    "flight": "F1",
                    "photo": "DJI_0001.JPG",
                    "photo_w": 1280,
                    "photo_h": 800,
                    "tiles": [
                        ("t1.png", 0, 0, 640, 640, 1),
                        ("t2.png", 320, 0, 640, 640, 2),
                        ("t3.png", 640, 0, 640, 640, 3),
                        ("t4.png", 0, 640, 640, 160, 4),
                    ],
                }
            ]
        )
        tiles = infer.read_tiles(manifest, root)
        model = FakeModel(
            {
                # A in t1 at full size, B distinct in t1, dup of A in t2, nothing in t3,
                # C half-cropped at the photo bottom border in t4 (tile_h < 640).
                1: [
                    {"xyxy": [380.0, 100.0, 460.0, 180.0], "conf": 0.9, "cls": 0},
                    {"xyxy": [20.0, 500.0, 100.0, 580.0], "conf": 0.8, "cls": 0},
                ],
                2: [{"xyxy": [60.0, 100.0, 140.0, 180.0], "conf": 0.7, "cls": 0}],
                3: [],
                4: [{"xyxy": [250.0, 140.0, 330.0, 160.0], "conf": 0.6, "cls": 0}],
            }
        )
        result = infer.count_photo(tiles, infer.load_image, model.predict, 0.25, 0.5)
        assert result.box_count == 4
        assert result.global_count == 3  # A merged; B and clipped C each counted once
        assert result.dedup_removed == 1
        assert result.source_tiles == ("t1.png", "t2.png", "t4.png")

    def test_single_tile_photo_skips_dedup(self, make_tile_input):
        manifest, root = make_tile_input(
            [
                {
                    "flight": "F3",
                    "photo": "DJI_0003.JPG",
                    "photo_w": 640,
                    "photo_h": 640,
                    "tiles": [("t1.png", 0, 0, 640, 640, 21)],
                }
            ]
        )
        tiles = infer.read_tiles(manifest, root)
        model = FakeModel(
            {
                # two heavily overlapping boxes (IoU ~0.68 >= 0.5): would merge
                # if dedup ran — with a single tile there is nothing to dedup
                # against, so both are counted (D6, FR-2 single-photo scenario).
                21: [
                    {"xyxy": [50.0, 50.0, 150.0, 150.0], "conf": 0.9, "cls": 0},
                    {"xyxy": [60.0, 60.0, 160.0, 160.0], "conf": 0.8, "cls": 0},
                ],
            }
        )
        result = infer.count_photo(tiles, infer.load_image, model.predict, 0.25, 0.5)
        assert result.global_count == 2
        assert result.box_count == 2
        assert result.dedup_removed == 0
        assert result.source_tiles == ("t1.png",)

    def test_sub_threshold_and_other_class_boxes_never_count(self, make_tile_input):
        manifest, root = make_tile_input(
            [
                {
                    "flight": "F1",
                    "photo": "DJI_0001.JPG",
                    "photo_w": 640,
                    "photo_h": 640,
                    "tiles": [("t1.png", 0, 0, 640, 640, 31)],
                }
            ]
        )
        tiles = infer.read_tiles(manifest, root)
        model = FakeModel(
            {31: [{"xyxy": [0.0, 0.0, 10.0, 10.0], "conf": 0.1, "cls": 0}]}
        )
        result = infer.count_photo(tiles, infer.load_image, model.predict, 0.25, 0.5)
        assert result.global_count == 0  # below --conf, dropped by the filter

    def test_empty_tiles_raise_value_error(self):
        with pytest.raises(ValueError):
            infer.count_photo([], infer.load_image, lambda img: [], 0.25, 0.5)


# ---------------------------------------------------------------------------
# PR2 2.6: render_csv — deterministic CSV text (FR-5, NFR-3)
# ---------------------------------------------------------------------------


class TestRenderCsv:
    def test_exact_header_and_byte_exact_row(self):
        rows = [
            {
                "flight": "F1",
                "photo": "a.JPG",
                "global_count": 2,
                "box_count": 3,
                "dedup_removed": 1,
                "source_tiles": ("t1.png", "t2.png"),
            }
        ]
        expected = (
            "flight,photo,global_count,box_count,dedup_removed,source_tiles\n"
            "F1,a.JPG,2,3,1,t1.png;t2.png\n"
        )
        assert infer.render_csv(rows) == expected

    def test_rows_are_sorted_by_flight_then_photo(self):
        rows = [
            {"flight": "B", "photo": "b2.JPG", "global_count": 1, "box_count": 1, "dedup_removed": 0, "source_tiles": ("x.png",)},
            {"flight": "A", "photo": "a2.JPG", "global_count": 4, "box_count": 4, "dedup_removed": 0, "source_tiles": ("y.png",)},
            {"flight": "A", "photo": "a1.JPG", "global_count": 2, "box_count": 2, "dedup_removed": 0, "source_tiles": ("z.png",)},
        ]
        lines = infer.render_csv(rows).strip().split("\n")[1:]
        assert lines[0].startswith("A,a1.JPG,")
        assert lines[1].startswith("A,a2.JPG,")
        assert lines[2].startswith("B,b2.JPG,")

    def test_semicolon_join_is_deterministic(self):
        row = {"flight": "F", "photo": "p.JPG", "global_count": 1, "box_count": 1, "dedup_removed": 0, "source_tiles": ("t2.png", "t1.png")}
        csv_text = infer.render_csv([row])
        assert "t2.png;t1.png" in csv_text  # join preserves the tuple order
        assert csv_text == infer.render_csv([row])  # byte-identical rerun

    def test_empty_rows_renders_header_only(self):
        assert infer.render_csv([]) == "flight,photo,global_count,box_count,dedup_removed,source_tiles\n"


# ---------------------------------------------------------------------------
# PR2 2.7: render_summaries — pinned per-flight summary lines (finding 6)
# ---------------------------------------------------------------------------


class TestRenderSummaries:
    def test_pinned_line_format_with_photo_counts_and_plant_sums(self):
        rows = [
            {"flight": "B", "photo": "b1.JPG", "global_count": 2, "box_count": 2, "dedup_removed": 0, "source_tiles": ()},
            {"flight": "A", "photo": "a1.JPG", "global_count": 1, "box_count": 1, "dedup_removed": 0, "source_tiles": ()},
            {"flight": "A", "photo": "a2.JPG", "global_count": 4, "box_count": 4, "dedup_removed": 0, "source_tiles": ()},
        ]
        assert infer.render_summaries(rows) == [
            "summary flight=A photos=2 plants=5",
            "summary flight=B photos=1 plants=2",
        ]

    def test_flight_order_is_first_seen_over_sorted_rows(self):
        rows = [
            {"flight": "Z", "photo": "z.JPG", "global_count": 1, "box_count": 1, "dedup_removed": 0, "source_tiles": ()},
            {"flight": "A", "photo": "a.JPG", "global_count": 1, "box_count": 1, "dedup_removed": 0, "source_tiles": ()},
        ]
        lines = infer.render_summaries(rows)
        assert lines[0].startswith("summary flight=A")
        assert lines[1].startswith("summary flight=Z")

    def test_empty_rows_yield_no_lines(self):
        assert infer.render_summaries([]) == []


# ---------------------------------------------------------------------------
# PR2 2.8: gate — hand-computed counts, byte-identical rerun, ultralytics-free
# ---------------------------------------------------------------------------


class TestPr2Gate:
    def test_pipeline_counts_match_hand_computed_values(self, make_tile_input):
        # End-to-end: manifest -> tiles -> FakeModel inference -> render_csv.
        manifest, root = make_tile_input(
            [
                {
                    "flight": "F2",
                    "photo": "DJI_0002.JPG",
                    "photo_w": 960,
                    "photo_h": 640,
                    "tiles": [("t1.png", 0, 0, 640, 640, 11), ("t2.png", 320, 0, 640, 640, 12)],
                }
            ]
        )
        tiles = infer.read_tiles(manifest, root)
        model = FakeModel(
            {
                11: [{"xyxy": [400.0, 200.0, 480.0, 280.0], "conf": 0.9, "cls": 0}],
                12: [{"xyxy": [80.0, 200.0, 160.0, 280.0], "conf": 0.8, "cls": 0}],
            }
        )
        result = infer.count_photo(tiles, infer.load_image, model.predict, 0.25, 0.5)
        row = {
            "flight": tiles[0].flight,
            "photo": tiles[0].photo,
            "global_count": result.global_count,
            "box_count": result.box_count,
            "dedup_removed": result.dedup_removed,
            "source_tiles": result.source_tiles,
        }
        assert infer.render_csv([row]) == (
            "flight,photo,global_count,box_count,dedup_removed,source_tiles\n"
            "F2,DJI_0002.JPG,1,2,1,t1.png;t2.png\n"
        )

    def test_render_is_byte_identical_on_rerun(self, make_tile_input):
        manifest, root = make_tile_input(
            [
                {
                    "flight": "F1",
                    "photo": "DJI_0001.JPG",
                    "photo_w": 1280,
                    "photo_h": 800,
                    "tiles": [("t1.png", 0, 0, 640, 640, 1), ("t2.png", 320, 0, 640, 640, 2)],
                }
            ]
        )
        tiles = infer.read_tiles(manifest, root)
        model = FakeModel({1: [{"xyxy": [380.0, 100.0, 460.0, 180.0], "conf": 0.9, "cls": 0}]})
        first = infer.count_photo(tiles, infer.load_image, model.predict, 0.25, 0.5)
        second = infer.count_photo(tiles, infer.load_image, model.predict, 0.25, 0.5)
        assert first == second  # count outcome is deterministic (NFR-2/3)
        row = {
            "flight": tiles[0].flight,
            "photo": tiles[0].photo,
            "global_count": first.global_count,
            "box_count": first.box_count,
            "dedup_removed": first.dedup_removed,
            "source_tiles": first.source_tiles,
        }
        assert infer.render_csv([row]) == infer.render_csv([row])

    def test_ultralytics_imports_only_inside_load_model(self):
        # infer.py: the runtime is NAMED (PR3 seam) but can only be imported
        # lazily inside load_model — never at module level (NFR-1); conftest
        # keeps the same strict rule as before (no imports of ultralytics).
        src = Path(infer.__file__).read_text(encoding="utf-8")
        for line in src.splitlines():
            if line.lstrip().startswith(("import ultralytics", "from ultralytics")):
                assert line.startswith("    ")  # inside a function, lazy
        conftest_src = Path(__file__).parent.joinpath("conftest.py").read_text(encoding="utf-8")
        for line in conftest_src.splitlines():
            assert not line.lstrip().startswith(("import ultralytics", "from ultralytics"))


# ---------------------------------------------------------------------------
# PR3 3.1: load_model — the ONLY ultralytics seam (lazy import inside) + the
# _UltralyticsPredictor adapter mapping results[0].boxes to pixel-xyxy dicts.
# ---------------------------------------------------------------------------


class _TensorShim:
    """Minimal fake for an ultralytics tensor: only what the adapter uses."""

    def __init__(self, data):
        self._data = data

    def tolist(self):
        return self._data


class _FakeBoxes:
    """results[0].boxes stand-in: xyxy/conf/cls as tensor shims."""

    def __init__(self, raw):
        # raw: list of (xyxy, conf, cls) tuples
        self.xyxy = _TensorShim([b[0] for b in raw])
        self.conf = _TensorShim([b[1] for b in raw])
        self.cls = _TensorShim([b[2] for b in raw])


class _FakeResults:
    def __init__(self, raw):
        self.boxes = _FakeBoxes(raw)


class _FakePredictorModel:
    """Model whose .predict returns canned ultralytics-style results."""

    def __init__(self, results):
        self._results = results

    def predict(self, image):
        return self._results


class TestUltralyticsAdapter:
    def test_maps_results_boxes_to_pixel_xyxy_dicts(self):
        raw = [((10.0, 20.0, 30.0, 40.0), 0.9, 0)]
        predictor = infer._UltralyticsPredictor(
            _FakePredictorModel([_FakeResults(raw)])
        )
        out = predictor.predict(np.zeros((4, 4, 3), dtype=np.uint8))
        assert out == [{"xyxy": [10.0, 20.0, 30.0, 40.0], "conf": 0.9, "cls": 0}]

    def test_maps_multiple_boxes_deterministically(self):
        raw = [
            ((0.0, 0.0, 10.0, 10.0), 0.9, 0),
            ((100.0, 100.0, 200.0, 200.0), 0.4, 0),
        ]
        predictor = infer._UltralyticsPredictor(
            _FakePredictorModel([_FakeResults(raw)])
        )
        out = predictor.predict(np.zeros((4, 4, 3), dtype=np.uint8))
        assert out == [
            {"xyxy": [0.0, 0.0, 10.0, 10.0], "conf": 0.9, "cls": 0},
            {"xyxy": [100.0, 100.0, 200.0, 200.0], "conf": 0.4, "cls": 0},
        ]

    def test_zero_boxes_yield_empty_list(self):
        predictor = infer._UltralyticsPredictor(
            _FakePredictorModel([_FakeResults([])])
        )
        assert predictor.predict(np.zeros((4, 4, 3), dtype=np.uint8)) == []

    def test_tolerates_singleton_nested_conf_and_cls_tensors(self):
        # ultralytics occasionally emits (N,1) tensors for conf/cls; the
        # adapter must unwrap one nesting level instead of producing [[0.9]].
        raw = [((10.0, 20.0, 30.0, 40.0), [0.9], [0])]
        predictor = infer._UltralyticsPredictor(
            _FakePredictorModel([_FakeResults(raw)])
        )
        out = predictor.predict(np.zeros((4, 4, 3), dtype=np.uint8))
        assert out[0]["conf"] == 0.9
        assert out[0]["cls"] == 0

    def test_fails_fast_when_results_have_no_boxes(self):
        class _NoBoxes:
            pass

        predictor = infer._UltralyticsPredictor(_FakePredictorModel([_NoBoxes()]))
        with pytest.raises(RuntimeError):
            predictor.predict(np.zeros((4, 4, 3), dtype=np.uint8))

    def test_fails_fast_when_results_are_empty(self):
        predictor = infer._UltralyticsPredictor(_FakePredictorModel([]))
        with pytest.raises(RuntimeError):
            predictor.predict(np.zeros((4, 4, 3), dtype=np.uint8))


class TestLoadModel:
    def test_imports_ultralytics_lazily_and_returns_a_predictor(self, monkeypatch):
        import types

        class _FakeYOLO:
            def __init__(self, weights):
                self.weights = weights

            def predict(self, image):
                # one Results per image, even with zero boxes (ultralytics shape)
                return [_FakeResults([])]

        fake_module = types.ModuleType("ultralytics")
        fake_module.YOLO = _FakeYOLO
        monkeypatch.setitem(sys.modules, "ultralytics", fake_module)

        predictor = infer.load_model("runs/train/exp/weights/best.pt")
        assert isinstance(predictor, infer._UltralyticsPredictor)
        assert predictor.predict(np.zeros((2, 2, 3), dtype=np.uint8)) == []

    def test_module_import_never_preloads_ultralytics(self):
        # NFR-1: importing infer (already imported at session start) must not
        # have pulled the runtime in — the seam is lazy by construction.
        assert "ultralytics" not in sys.modules
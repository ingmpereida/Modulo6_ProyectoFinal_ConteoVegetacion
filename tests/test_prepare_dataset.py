"""Tests for prepare_dataset.py pure core and CLI (FR-1..FR-4, NFR-1).

Runs CPU-only with synthetic fixtures (FR-6, NFR-4); no GPU or ultralytics
runtime is required.
"""

import csv
from pathlib import Path

import pytest
import yaml

import prepare_dataset as pd


def manifest_rows(manifest: Path) -> list[dict]:
    """Parse the synthetic manifest the same shape read_manifest produces."""
    with manifest.open(newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


# ---------------------------------------------------------------------------
# FR-1: group_flights
# ---------------------------------------------------------------------------


class TestGroupFlights:
    def test_three_flights_produce_three_groups(self, three_flights):
        manifest, _ = three_flights
        groups = pd.group_flights(manifest_rows(manifest))

        assert set(groups.keys()) == {
            "ParcelaA_2026-09-10_emergencia",
            "ParcelaB_2026-09-12_emergencia",
            "ParcelaC_2026-09-14_emergencia",
        }
        assert [len(rows) for rows in groups.values()] == [10, 5, 1]
        # Every row lands under its own flight, none lost.
        all_tiles = [tile for rows in groups.values() for tile in (r["tile"] for r in rows)]
        assert len(all_tiles) == 16

    def test_single_flight_produces_one_group(self, single_flight):
        manifest, _ = single_flight
        groups = pd.group_flights(manifest_rows(manifest))

        assert len(groups) == 1
        only = next(iter(groups.values()))
        assert len(only) == 5
        assert all(r["vuelo"] == "ParcelaA_2026-09-10_emergencia" for r in only)

    def test_interleaved_rows_merge_into_one_group_preserving_order(self):
        rows = [
            {"vuelo": "A", "tile": "t1.jpg"},
            {"vuelo": "B", "tile": "t2.jpg"},
            {"vuelo": "A", "tile": "t3.jpg"},
            {"vuelo": "C", "tile": "t4.jpg"},
            {"vuelo": "B", "tile": "t5.jpg"},
        ]
        groups = pd.group_flights(rows)

        assert list(groups.keys()) == ["A", "B", "C"]
        assert [r["tile"] for r in groups["A"]] == ["t1.jpg", "t3.jpg"]
        assert [r["tile"] for r in groups["B"]] == ["t2.jpg", "t5.jpg"]
        assert [r["tile"] for r in groups["C"]] == ["t4.jpg"]


# ---------------------------------------------------------------------------
# FR-2 / NFR-1: split_tiles
# ---------------------------------------------------------------------------


class TestSplitTiles:
    TEN_TILES = [f"DJI_0001_x{x:05d}_y00000.jpg" for x in range(10)]

    def test_ten_tiles_split_eight_train_two_valid(self):
        train, valid = pd.split_tiles(self.TEN_TILES, valid_ratio=0.2, seed=42)
        assert len(train) == 8
        assert len(valid) == 2

    def test_same_seed_twice_yields_identical_sets(self):
        first = pd.split_tiles(self.TEN_TILES, valid_ratio=0.2, seed=42)
        second = pd.split_tiles(self.TEN_TILES, valid_ratio=0.2, seed=42)
        assert first == second

    def test_single_tile_flight_puts_all_in_train(self):
        train, valid = pd.split_tiles(["DJI_0001_x00000_y00000.jpg"], seed=42)
        assert train == ["DJI_0001_x00000_y00000.jpg"]
        assert valid == []

    def test_two_tiles_split_one_and_one(self):
        tiles = ["DJI_0001_x00000_y00000.jpg", "DJI_0001_x00544_y00000.jpg"]
        train, valid = pd.split_tiles(tiles, valid_ratio=0.2, seed=42)
        assert len(train) == 1
        assert len(valid) == 1
        assert train + valid == sorted(tiles)

    def test_half_ratio_splits_ten_five_and_five(self):
        train, valid = pd.split_tiles(self.TEN_TILES, valid_ratio=0.5, seed=42)
        assert len(train) == 5
        assert len(valid) == 5

    def test_no_tile_is_lost_or_duplicated(self):
        train, valid = pd.split_tiles(self.TEN_TILES, valid_ratio=0.2, seed=7)
        combined = sorted(train + valid)
        assert combined == sorted(self.TEN_TILES)

    def test_input_order_does_not_affect_split(self):
        shuffled = list(reversed(self.TEN_TILES))
        train_a, valid_a = pd.split_tiles(self.TEN_TILES, valid_ratio=0.2, seed=42)
        train_b, valid_b = pd.split_tiles(shuffled, valid_ratio=0.2, seed=42)
        assert sorted(train_a) == sorted(train_b)
        assert sorted(valid_a) == sorted(valid_b)


# ---------------------------------------------------------------------------
# FR-3 / NFR-1: build_yaml
# ---------------------------------------------------------------------------


class TestBuildYaml:
    def test_default_yaml_round_trips_with_plant_names(self, tmp_path):
        dataset = tmp_path / "dataset"
        data = pd.build_yaml(dataset)

        dumped = yaml.safe_dump(data, sort_keys=False)
        parsed = yaml.safe_load(dumped)

        assert list(parsed.keys()) == ["path", "train", "val", "names"]
        assert parsed["path"] == str(dataset)
        assert parsed["train"] == "images/train"
        assert parsed["val"] == "images/valid"
        assert parsed["names"] == ["plant"]

    def test_custom_names_round_trip(self, tmp_path):
        data = pd.build_yaml(tmp_path / "dataset", names=["plant", "weed"])
        parsed = yaml.safe_load(yaml.safe_dump(data, sort_keys=False))
        assert parsed["names"] == ["plant", "weed"]

    def test_dump_is_deterministic_byte_for_byte(self, tmp_path):
        data = pd.build_yaml(tmp_path / "dataset")
        first = yaml.safe_dump(data, sort_keys=False)
        second = yaml.safe_dump(data, sort_keys=False)
        assert first == second


# ---------------------------------------------------------------------------
# FR-3 / C-3: copy_label (the single label-touching seam)
# ---------------------------------------------------------------------------


class TestCopyLabel:
    def test_valid_sidecar_is_copied_with_content(self, tmp_path):
        src = tmp_path / "DJI_0001_x00000_y00000.txt"
        src.write_text("0 0.5 0.5 0.2 0.2\n")
        dst_dir = tmp_path / "labels"
        dst_dir.mkdir()

        result = pd.copy_label(src, dst_dir)

        assert result == dst_dir / src.name
        assert result.read_text() == "0 0.5 0.5 0.2 0.2\n"

    def test_empty_sidecar_is_copied_as_empty_label(self, tmp_path):
        # An empty YOLO sidecar means "image with no objects" — valid, not background.
        src = tmp_path / "tile.txt"
        src.write_text("")
        dst_dir = tmp_path / "labels"
        dst_dir.mkdir()

        result = pd.copy_label(src, dst_dir)

        assert result is not None
        assert result.read_text() == ""

    def test_missing_sidecar_returns_none_for_background(self, tmp_path):
        dst_dir = tmp_path / "labels"
        dst_dir.mkdir()

        result = pd.copy_label(tmp_path / "absent.txt", dst_dir)

        assert result is None
        assert list(dst_dir.iterdir()) == []

    def test_malformed_short_line_raises_hard_error(self, tmp_path):
        src = tmp_path / "bad.txt"
        src.write_text("0 0.5 0.5\n")  # 3 fields, not 5
        dst_dir = tmp_path / "labels"
        dst_dir.mkdir()

        with pytest.raises(pd.LabelFormatError):
            pd.copy_label(src, dst_dir)
        assert list(dst_dir.iterdir()) == []  # nothing copied

    def test_malformed_non_numeric_field_raises_hard_error(self, tmp_path):
        src = tmp_path / "bad.txt"
        src.write_text("0 0.5 abc 0.2 0.2\n")
        dst_dir = tmp_path / "labels"
        dst_dir.mkdir()

        with pytest.raises(pd.LabelFormatError):
            pd.copy_label(src, dst_dir)
"""Tests for prepare_dataset.py pure core and CLI (FR-1..FR-4, NFR-1).

Runs CPU-only with synthetic fixtures (FR-6, NFR-4); no GPU or ultralytics
runtime is required.
"""

import csv
from pathlib import Path

import pytest

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
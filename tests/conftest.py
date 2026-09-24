"""Shared pytest fixtures for the YOLO training pipeline suite.

Fixtures build synthetic tile layouts the same way tile_pipeline.py does:
per-flight folders under a tiles root, tile files named <foto>_x{5}_y{5}.jpg,
optional YOLO .txt sidecars next to each image, and a manifest.csv recording
every tile. CPU-only: image files are dummy bytes, never decoded, and no
GPU/ultralytics runtime is required (FR-6, NFR-4).
"""

import csv
from pathlib import Path

import pytest

# Columns written by tile_pipeline.py (see its manifest.csv contract).
MANIFEST_FIELDS = [
    "vuelo",
    "imagen_origen",
    "tile",
    "x_offset",
    "y_offset",
    "tile_w",
    "tile_h",
    "imagen_ancho",
    "imagen_alto",
]

# Placeholder bytes; prepare_dataset.py copies images without decoding them.
FAKE_JPEG = b"\xff\xd8\xff\xe0fake-jpeg-bytes"

# Normalized YOLO sidecar content (class x y w h).
VALID_LABEL = "0 0.5 0.5 0.2 0.2\n"


def tile_name(foto: str, x: int, y: int = 0) -> str:
    """Produce a tile filename matching tile_pipeline.py's <foto>_x{5}_y{5}.jpg scheme."""
    return f"{foto}_x{x:05d}_y{y:05d}.jpg"


@pytest.fixture
def make_flight_layout(tmp_path: Path):
    """Factory building a synthetic tiles root plus manifest.csv.

    flights: dict[vuelo -> dict[tile_name -> label_text | None]]
    A None label means the sidecar is missing (background tile, FR-3).
    Returns a (manifest_path, labels_root) tuple.
    """

    def _make(flights: dict, manifest_name: str = "manifest.csv"):
        labels_root = tmp_path / "tiles"
        rows = []
        for vuelo, tiles in flights.items():
            flight_dir = labels_root / vuelo
            flight_dir.mkdir(parents=True, exist_ok=True)
            for tile, label_text in tiles.items():
                (flight_dir / tile).write_bytes(FAKE_JPEG)
                if label_text is not None:
                    (flight_dir / tile.replace(".jpg", ".txt")).write_text(label_text)
                rows.append(
                    {
                        "vuelo": vuelo,
                        "imagen_origen": "DJI_0001.JPG",
                        "tile": tile,
                        "x_offset": 0,
                        "y_offset": 0,
                        "tile_w": 640,
                        "tile_h": 640,
                        "imagen_ancho": 4000,
                        "imagen_alto": 3000,
                    }
                )
        manifest = tmp_path / manifest_name
        with manifest.open("w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=MANIFEST_FIELDS)
            writer.writeheader()
            writer.writerows(rows)
        return manifest, labels_root

    return _make


@pytest.fixture
def three_flights(make_flight_layout):
    """Three flights: 10 labeled tiles, 4 labeled + 1 background, and a single-tile edge flight."""
    flight_b = {tile_name("DJI_0002", x): VALID_LABEL for x in range(4)}
    flight_b[tile_name("DJI_0003", 0)] = None  # background tile
    flights = {
        "ParcelaA_2026-09-10_emergencia": {
            tile_name("DJI_0001", x): VALID_LABEL for x in range(10)
        },
        "ParcelaB_2026-09-12_emergencia": flight_b,
        "ParcelaC_2026-09-14_emergencia": {tile_name("DJI_0004", 0): VALID_LABEL},
    }
    return make_flight_layout(flights)


@pytest.fixture
def single_flight(make_flight_layout):
    """One flight with five labeled tiles."""
    flights = {
        "ParcelaA_2026-09-10_emergencia": {
            tile_name("DJI_0001", x): VALID_LABEL for x in range(5)
        }
    }
    return make_flight_layout(flights)
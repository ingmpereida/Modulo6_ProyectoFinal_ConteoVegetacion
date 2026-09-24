"""Tests for convert_polygon_to_bbox.py.

Covers the Roboflow polygon-YOLO to bbox-YOLO converter: polygon->bbox math
and precision, tile-name restoration across jpg/jpeg/png variants, the
exact 5-field downstream contract, validate-then-write failure modes
(malformed label, odd coordinate count, zero-area polygon, empty export),
missing-sidecar handling (background tile), determinism, and the CLI
end-to-end behavior. CPU-only and ultralytics-free — pure stdlib logic.
"""

import subprocess
import sys

import pytest

import convert_polygon_to_bbox as conv


BOX_PRECISION = 8


def write_sidecar(labels_dir, stem, lines):
    (labels_dir / f"{stem}.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


# ---------------------------------------------------------------------------
# roboflow_stem_to_tile: filename restoration
# ---------------------------------------------------------------------------


class TestRoboflowStemToTile:
    def test_strips_jpg_marker_and_keeps_tile_name(self):
        assert (
            conv.roboflow_stem_to_tile(
                "Parcela01_2026-09-23_etapa_x00000_y00000_jpg.rf.23156e5a.jpg"
            )
            == "Parcela01_2026-09-23_etapa_x00000_y00000"
        )

    def test_strips_jpeg_marker(self):
        assert (
            conv.roboflow_stem_to_tile("tile_01_jpeg.rf.abc123.jpeg")
            == "tile_01"
        )

    def test_strips_png_marker(self):
        assert conv.roboflow_stem_to_tile("tile_02_png.rf.abc123.png") == "tile_02"

    def test_keeps_name_without_marker(self):
        assert conv.roboflow_stem_to_tile("plain_name.jpg") == "plain_name"


# ---------------------------------------------------------------------------
# polygon_to_bbox: math, precision, and failure modes
# ---------------------------------------------------------------------------


class TestPolygonToBbox:
    def test_bbox_from_polygon_minmax(self):
        # Square: (0.1,0.2) (0.5,0.2) (0.5,0.6) (0.1,0.6) -> cx .3, cy .4, w .4, h .4
        out = conv.polygon_to_bbox("0", ["0.1", "0.2", "0.5", "0.2", "0.5", "0.6", "0.1", "0.6"])
        parts = out.split()
        assert len(parts) == 5
        assert parts[0] == "0"
        assert float(parts[1]) == pytest.approx(0.3)
        assert float(parts[2]) == pytest.approx(0.4)
        assert float(parts[3]) == pytest.approx(0.4)
        assert float(parts[4]) == pytest.approx(0.4)

    def test_eight_decimal_precision_and_parseable(self):
        out = conv.polygon_to_bbox("0", ["0.123456789", "0.2", "0.5", "0.600000001"])
        parts = out.split()
        for tok in parts[1:]:
            assert len(tok.split(".")[1]) == BOX_PRECISION
            float(tok)  # must not raise

    def test_odd_coordinate_count_raises(self):
        # F2: an odd x with no matching y must fail loudly, never corrupt.
        with pytest.raises(conv.LabelFormatError, match="debe ser par"):
            conv.polygon_to_bbox("0", ["0.1", "0.2", "0.5"])

    def test_non_numeric_coordinate_raises(self):
        with pytest.raises(conv.LabelFormatError, match="no numerica"):
            conv.polygon_to_bbox("0", ["0.1", "abc", "0.5", "0.6"])

    def test_zero_area_polygon_raises(self):
        # F5: a degenerate polygon produces a junk zero-size box for YOLO.
        with pytest.raises(conv.LabelFormatError, match="degenerado"):
            conv.polygon_to_bbox("0", ["0.5", "0.5", "0.5", "0.5"])


# ---------------------------------------------------------------------------
# convert_label and the 5-field downstream contract
# ---------------------------------------------------------------------------


class TestConvertLabel:
    def test_writes_bbox_lines_with_trailing_newline(self, tmp_path):
        write_sidecar(tmp_path, "tile", ["0 0.1 0.2 0.5 0.2 0.5 0.6 0.1 0.6", "0 0.0 0.0 1.0 1.0"])
        out = tmp_path / "tile.txt"
        count = conv.convert_label(tmp_path / "tile.txt", out)
        assert count == 2
        lines = out.read_text(encoding="utf-8").splitlines()
        assert len(lines) == 2
        for line in lines:
            fields = line.split()
            assert len(fields) == 5  # downstream contract: exactly 5 fields
            assert fields[0].isdigit()
            for tok in fields[1:]:
                float(tok)

    def test_skips_blank_lines(self, tmp_path):
        write_sidecar(tmp_path, "tile", ["", "0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8", "  "])
        out = tmp_path / "tile.txt"
        count = conv.convert_label(tmp_path / "tile.txt", out)
        assert count == 1

    def test_empty_sidecar_writes_empty_file(self, tmp_path):
        write_sidecar(tmp_path, "tile", [])
        out = tmp_path / "tile.txt"
        count = conv.convert_label(tmp_path / "tile.txt", out)
        assert count == 0
        assert out.read_text(encoding="utf-8") == ""

    def test_malformed_line_fails_with_file_and_line_context(self, tmp_path):
        # F1: fail with context, never a raw traceback with no location.
        write_sidecar(tmp_path, "tile", ["0 0.1 0.2 0.5", "0 0.9 0.9 0.9 0.9"])
        with pytest.raises(conv.LabelFormatError) as exc:
            conv.convert_label(tmp_path / "tile.txt", tmp_path / "out.txt")
        assert "tile.txt:1" in str(exc.value)


# ---------------------------------------------------------------------------
# collect_tiles: empty export must fail, not fake success
# ---------------------------------------------------------------------------


class TestCollectTiles:
    def test_no_images_raises_and_never_reports_success(self, tmp_path):
        # F4: a typo'd export path must not print "0 tiles" with exit 0.
        images = tmp_path / "images"
        images.mkdir(parents=True)
        with pytest.raises(conv.LabelFormatError, match="no se encontraron imagenes"):
            conv.collect_tiles(tmp_path)

    def test_globs_all_supported_extensions(self, tmp_path):
        images = tmp_path / "images"
        images.mkdir(parents=True)
        (images / "a.jpg").touch()
        (images / "b.jpeg").touch()
        (images / "c.png").touch()
        found = conv.collect_tiles(tmp_path)
        assert [p.name for p in found] == ["a.jpg", "b.jpeg", "c.png"]


# ---------------------------------------------------------------------------
# main(): validate-then-write end-to-end + determinism
# ---------------------------------------------------------------------------


def build_export(tmp_path):
    """Creates an export tree: 2 labeled jpg tiles + 1 background tile."""
    export = tmp_path / "export" / "train"
    (export / "images").mkdir(parents=True)
    (export / "labels").mkdir(parents=True)
    (export / "images" / "Parcela01_x0_y0_jpg.rf.aaa.jpg").write_bytes(b"IMG1")
    (export / "images" / "Parcela01_x0_y1_jpg.rf.bbb.jpg").write_bytes(b"IMG2")
    (export / "images" / "Parcela02_x0_y0_jpg.rf.ccc.jpg").write_bytes(b"IMG3")
    write_sidecar(
        export / "labels",
        "Parcela01_x0_y0_jpg.rf.aaa",
        ["0 0.1 0.2 0.5 0.2 0.5 0.6 0.1 0.6"],
    )
    write_sidecar(
        export / "labels",
        "Parcela01_x0_y1_jpg.rf.bbb",
        ["0 0.0 0.0 1.0 1.0"],
    )
    return export


class TestMain:
    def test_converts_labels_and_restores_tile_names(self, tmp_path):
        export = build_export(tmp_path)
        out = tmp_path / "out"
        rc = conv.main(["--export", str(export), "--output", str(out)])
        assert rc == 0
        # Names restored, images copied with correct extension.
        assert (out / "Parcela01_x0_y0.jpg").read_bytes() == b"IMG1"
        assert (out / "Parcela01_x0_y1.jpg").read_bytes() == b"IMG2"
        assert (out / "Parcela02_x0_y0.jpg").read_bytes() == b"IMG3"
        # 2 sidecars converted; background tile has no sidecar.
        assert (out / "Parcela01_x0_y0.txt").exists()
        assert (out / "Parcela01_x0_y1.txt").exists()
        assert not (out / "Parcela02_x0_y0.txt").exists()
        # No Roboflow marker leaks into output names.
        assert not list(out.glob("*_rf.*"))

    def test_preserves_png_extension(self, tmp_path):
        export = build_export(tmp_path)
        (export / "images" / "T3_png.rf.fff.png").write_bytes(b"PNGDATA")
        out = tmp_path / "out"
        conv.main(["--export", str(export), "--output", str(out)])
        assert (out / "T3.png").read_bytes() == b"PNGDATA"

    def test_malformed_label_writes_nothing(self, tmp_path):
        export = build_export(tmp_path)
        (export / "labels" / "Parcela02_x0_y0_jpg.rf.ccc.txt").write_text(
            "0 0.1 0.2 0.3\n", encoding="utf-8"
        )
        out = tmp_path / "out"
        rc = conv.main(["--export", str(export), "--output", str(out)])
        assert rc == 1
        # Validate-then-write: nothing at all was written.
        assert not out.exists() or not any(out.iterdir())

    def test_missing_export_fails_with_exit_1(self, tmp_path):
        rc = conv.main(["--export", str(tmp_path / "nope"), "--output", str(tmp_path / "out")])
        assert rc == 1

    def test_deterministic_output(self, tmp_path):
        export = build_export(tmp_path)
        out1, out2 = tmp_path / "o1", tmp_path / "o2"
        conv.main(["--export", str(export), "--output", str(out1)])
        conv.main(["--export", str(export), "--output", str(out2)])
        names1 = sorted(p.name for p in out1.iterdir())
        names2 = sorted(p.name for p in out2.iterdir())
        assert names1 == names2
        for name in names1:
            assert (out1 / name).read_bytes() == (out2 / name).read_bytes()

    def test_cli_subprocess_runs_end_to_end(self, tmp_path):
        export = build_export(tmp_path)
        out = tmp_path / "out"
        result = subprocess.run(
            [sys.executable, "convert_polygon_to_bbox.py", "--export", str(export), "--output", str(out)],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, result.stderr
        assert "3 tiles, 2 cajas" in result.stdout
        assert (out / "Parcela01_x0_y0.jpg").exists()
#!/usr/bin/env python3
"""
convert_polygon_to_bbox.py
==========================

Convierte etiquetas YOLO en formato POLIGONO (clase x1 y1 x2 y2 ...) al
formato CAJA que consume el pipeline (clase cx cy w h, 5 campos), y reconstruye
el nombre original del tile desde el renombrado que aplica Roboflow al exportar.

Uso tipico (tras exportar desde Roboflow en formato yolov8):

    python3 convert_polygon_to_bbox.py \
        --export datasets/roboflow_export/train \
        --output datasets/labels_src/fotos_dron

Entrada esperada (--export):

    train/
      images/<tile>_jpg.rf.<hash>.jpg      <- nombre Roboflow, un tile por imagen
      labels/<tile>_jpg.rf.<hash>.txt      <- poligonos YOLO: 0 x1 y1 x2 y2 ...

Salida (--output): el directorio de vuelo que consume prepare_dataset.py
(--labels), escribiendo por cada tile:

    <tile>.jpg                             <- imagen con el nombre original del tile
    <tile>.txt                             <- caja YOLO: 0 cx cy w h (bbox del poligono)

El formato de extension se conserva tal cual viene del export (jpg/jpeg/png),
y el marcador de Roboflow se detecta por la extension real del archivo. La
conversion es validate-then-write: si CUALQUIER label esta malformado, el
script falla con mensaje contextual y no escribe nada (misma filosofia
fail-fast que prepare_dataset.py).

Por que existe: Roboflow permite anotar con poligonos y su export 'yolov8'
los conserva como poligonos, pero prepare_dataset.py e infer.py exigen cajas
(class x y w h normalizadas). Este script cierra esa brecha convirtiendo cada
poligono a su bounding box (min/max de x e y).
"""

import argparse
import shutil
import sys
from pathlib import Path
from typing import Iterable

# Extensiones de imagen aceptadas en el export de Roboflow (misma familia que
# IMAGE_EXTENSIONS de tile_pipeline.py).
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}

# Precision de los floats normalizados que se escriben en los sidecars. Fija
# en 8 decimales: suficiente para coordenadas YOLO normalizadas (~0.4 px en un
# tile de 640x640) y mantiene el contrato de exactamente 5 campos por linea.
BBOX_PRECISION = 8


class LabelFormatError(ValueError):
    """Label malformado: el script falla con contexto y no escribe nada."""


def roboflow_stem_to_tile(name: str) -> str:
    """Restaura el nombre original del tile desde el nombre Roboflow.

    Roboflow reescribe el stem como "<tile>_<ext>.rf.<hash>.<ext>". Se quita la
    extension y luego el marcador "_<ext>.rf.", detectado por la extension real
    del archivo (jpg/jpeg/png).
    """
    stem = name
    suffix = Path(name).suffix.lower()
    if suffix in IMAGE_EXTENSIONS:
        stem = stem[: -len(suffix)]
    marker = f"_{suffix.lstrip('.')}.rf."
    if marker in stem:
        stem = stem.split(marker, 1)[0]
    return stem


def polygon_to_bbox(class_id: str, points: list[str]) -> str:
    """Convierte un poligono YOLO (class x1 y1 x2 y2 ...) a su bounding box.

    El poligono debe tener un numero PAR de coordenadas (pares x,y). Con
    coordenadas impares se lanza LabelFormatError: la ultima x quedaria sin su
    y y el rango vertical se colapsaria en silencio (corrupcion silenciosa).
    """
    if len(points) % 2 != 0:
        raise LabelFormatError(
            f"poligono con {len(points)} coordenadas (debe ser par): {class_id} {' '.join(points)}"
        )
    try:
        coords = [float(v) for v in points]
    except ValueError as exc:
        raise LabelFormatError(
            f"coordenada no numerica: {exc} (linea: {class_id} {' '.join(points)})"
        ) from None
    xs, ys = coords[0::2], coords[1::2]
    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(ys), max(ys)
    cx = (x_min + x_max) / 2
    cy = (y_min + y_max) / 2
    w = x_max - x_min
    h = y_max - y_min
    if w == 0 and h == 0:
        raise LabelFormatError(
            f"poligono degenerado (area cero): {class_id} {' '.join(points)}"
        )
    return f"{class_id} {cx:.{BBOX_PRECISION}f} {cy:.{BBOX_PRECISION}f} {w:.{BBOX_PRECISION}f} {h:.{BBOX_PRECISION}f}"


def parse_label(txt_path: Path) -> list[str]:
    """Parsea un sidecar de poligonos a cajas SIN escribir nada.

    Lineas vacias se ignoran; una linea malformada aborta con contexto de
    archivo:linea. Devuelve las lineas bbox ya formateadas. Separado de
    convert_label para poder validar TODO el export antes de escribir nada
    (validate-then-write).
    """
    lines = txt_path.read_text(encoding="utf-8").splitlines()
    bboxes = []
    for lineno, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        fields = line.split()
        try:
            bboxes.append(polygon_to_bbox(fields[0], fields[1:]))
        except LabelFormatError as exc:
            raise LabelFormatError(
                f"malformed label {txt_path}:{lineno}: {exc}"
            ) from None
    return bboxes


def convert_label(txt_path: Path, out_txt: Path) -> int:
    """Parsea y escribe un sidecar ya validado; devuelve la cantidad de cajas."""
    bboxes = parse_label(txt_path)
    out_txt.write_text("\n".join(bboxes) + ("\n" if bboxes else ""), encoding="utf-8")
    return len(bboxes)


def collect_tiles(export: Path) -> list[Path]:
    """Lista las imagenes del export Roboflow, fallando si no hay ninguna.

    Una ruta mal escrita (--export sin images/) no debe reportar "0 tiles"
    como exito: eso esconderia el error hasta la fase siguiente.
    """
    images = sorted(
        p for ext in IMAGE_EXTENSIONS for p in (export / "images").glob(f"*{ext}")
    )
    if not images:
        raise LabelFormatError(
            f"no se encontraron imagenes en {export / 'images'} "
            f"(extensiones {sorted(IMAGE_EXTENSIONS)})"
        )
    return images


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Convierte poligonos YOLO de Roboflow a cajas YOLO y restaura nombres de tile."
    )
    parser.add_argument("--export", required=True, help="carpeta train/ del export Roboflow (con images/ y labels/)")
    parser.add_argument("--output", required=True, help="directorio de vuelo de salida (el --labels de prepare_dataset.py)")
    args = parser.parse_args(argv)

    export = Path(args.export)
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)

    try:
        images = collect_tiles(export)
    except LabelFormatError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    # Fase 1 - validar TODO antes de escribir nada (validate-then-write).
    # Se parsean los sidecars pero NO se escribe; recien en fase 2, cuando
    # todo el export es valido, se copian imagenes y sidecars. Un label
    # malformado aborta con mensaje contextual y exit 1, sin escribir nada.
    parsed: dict[Path, list[str] | None] = {}
    try:
        for img in images:
            tile_stem = roboflow_stem_to_tile(img.name)
            src_txt = export / "labels" / (img.stem + ".txt")
            if src_txt.exists():
                parsed[img] = parse_label(src_txt)
            else:
                parsed[img] = None  # tile de fondo sin etiqueta
    except LabelFormatError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    # Fase 2 - escribir (recien aca se copian imagenes y sidecars).
    total_boxes = 0
    for img, bboxes in parsed.items():
        tile_stem = roboflow_stem_to_tile(img.name)
        suffix = img.suffix.lower()
        shutil.copyfile(img, out / f"{tile_stem}{suffix}")
        if bboxes is not None:
            out_txt = out / f"{tile_stem}.txt"
            out_txt.write_text("\n".join(bboxes) + ("\n" if bboxes else ""), encoding="utf-8")
            total_boxes += len(bboxes)
            print(f"{tile_stem}: {len(bboxes)} cajas")
        else:
            print(f"{tile_stem}: sin etiqueta (tile de fondo)")

    print(f"\nListo. {len(images)} tiles, {total_boxes} cajas en total -> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
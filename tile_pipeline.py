#!/usr/bin/env python3
"""
tile_pipeline.py
=================

Fase 2 - Preparacion de imagenes para etiquetado.

Corta fotos de dron (grandes, con plantas pequenas) en tiles (recortes)
mas pequenos, para que cada planta ocupe suficientes pixeles y sea facil
de etiquetar en Roboflow/CVAT. Tambien descarta automaticamente tiles
"vacios" (puro suelo uniforme, sin nada que etiquetar) para no perder
tiempo etiquetando recortes sin contenido util.

Estructura de entrada esperada (la misma que sugiere el checklist de campo):

    input_dir/
        ParcelaA_2026-09-10_emergencia/
            DJI_0001.JPG
            DJI_0002.JPG
            ...
        ParcelaB_2026-09-12_emergencia/
            DJI_0001.JPG
            ...

Cada subcarpeta = un vuelo (una parcela, una fecha, una etapa).

Salida generada:

    output_dir/
        ParcelaA_2026-09-10_emergencia/
            DJI_0001_x00000_y00000.jpg
            DJI_0001_x00544_y00000.jpg
            ...
        ParcelaB_2026-09-12_emergencia/
            ...
        manifest.csv   <- registro de cada tile: imagen origen, posicion, tamano

El manifest.csv es importante: mas adelante permite (a) saber de que foto
original vino cada tile, y (b) si se etiqueta a nivel de tile, reconstruir
o auditar el conteo total por foto o por parcela.

Uso basico:
    python3 tile_pipeline.py --input ./fotos_dron --output ./tiles

Uso con parametros ajustados:
    python3 tile_pipeline.py --input ./fotos_dron --output ./tiles \
        --tile-size 768 --overlap 0.15 --min-pixels 150

Requisitos:
    pip install pillow numpy --break-system-packages
"""

import argparse
import csv
import sys
from pathlib import Path

from PIL import Image
import numpy as np

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".tif", ".tiff"}


def find_flight_folders(input_dir: Path):
    """Cada subcarpeta directa de input_dir se trata como un vuelo (parcela+fecha+etapa).
    Si el usuario apunta directo a una carpeta con fotos sueltas (sin subcarpetas),
    esa carpeta se trata como un unico 'vuelo'."""
    subfolders = [f for f in input_dir.iterdir() if f.is_dir()]
    if subfolders:
        return sorted(subfolders)
    return [input_dir]


def list_images(folder: Path):
    return sorted(
        [f for f in folder.iterdir() if f.suffix.lower() in IMAGE_EXTENSIONS]
    )


def tile_positions(width, height, tile_size, overlap):
    """Genera las coordenadas (x, y) de la esquina superior-izquierda de cada tile,
    cubriendo toda la imagen con el traslape indicado. El ultimo tile de cada fila/
    columna se ajusta hacia adentro para no salirse del borde de la imagen."""
    stride = max(1, int(tile_size * (1 - overlap)))

    xs = list(range(0, max(width - tile_size, 0) + 1, stride))
    ys = list(range(0, max(height - tile_size, 0) + 1, stride))

    # Asegurar cobertura del borde derecho / inferior si la imagen no es multiplo exacto
    if not xs or xs[-1] + tile_size < width:
        xs.append(max(width - tile_size, 0))
    if not ys or ys[-1] + tile_size < height:
        ys.append(max(height - tile_size, 0))

    # Quitar duplicados manteniendo orden
    xs = sorted(set(xs))
    ys = sorted(set(ys))
    return [(x, y) for y in ys for x in xs]


def has_enough_detail(tile_img: Image.Image, min_pixels: int, color_diff: float = 25.0) -> bool:
    """Descarta tiles casi uniformes (ej. franjas de suelo desnudo sin nada que
    etiquetar), sin penalizar tiles con plantas pequenas y dispersas (comun en
    etapas de emergencia). En vez de medir la variacion total del tile -que es
    baja cuando las plantas cubren poca area aunque esten presentes-, cuenta
    cuantos pixeles se distinguen claramente del color de fondo dominante.
    min_pixels mas alto = mas estricto (descarta mas tiles casi vacios)."""
    arr = np.asarray(tile_img.convert("RGB"), dtype=np.int16)
    flat = arr.reshape(-1, 3)
    background = np.median(flat, axis=0)
    diff = np.abs(arr - background).sum(axis=2)
    foreground_pixels = int((diff > color_diff).sum())
    return foreground_pixels >= min_pixels


def process_flight_folder(folder: Path, out_root: Path, tile_size: int,
                           overlap: float, min_pixels: int, manifest_rows: list):
    images = list_images(folder)
    if not images:
        print(f"  (sin imagenes en {folder.name}, se omite)")
        return

    out_folder = out_root / folder.name
    out_folder.mkdir(parents=True, exist_ok=True)

    kept, skipped = 0, 0
    for img_path in images:
        try:
            img = Image.open(img_path).convert("RGB")
        except Exception as e:
            print(f"  ! No se pudo abrir {img_path.name}: {e}")
            continue

        w, h = img.size
        if w < tile_size or h < tile_size:
            print(f"  ! {img_path.name} es mas pequena que el tile ({w}x{h}), se copia entera")
            positions = [(0, 0)]
            effective_tile_w, effective_tile_h = w, h
        else:
            positions = tile_positions(w, h, tile_size, overlap)
            effective_tile_w, effective_tile_h = tile_size, tile_size

        for (x, y) in positions:
            box = (x, y, min(x + effective_tile_w, w), min(y + effective_tile_h, h))
            tile = img.crop(box)

            if not has_enough_detail(tile, min_pixels):
                skipped += 1
                continue

            tile_name = f"{img_path.stem}_x{x:05d}_y{y:05d}.jpg"
            tile.save(out_folder / tile_name, quality=95)

            manifest_rows.append({
                "vuelo": folder.name,
                "imagen_origen": img_path.name,
                "tile": tile_name,
                "x_offset": x,
                "y_offset": y,
                "tile_w": box[2] - box[0],
                "tile_h": box[3] - box[1],
                "imagen_ancho": w,
                "imagen_alto": h,
            })
            kept += 1

    print(f"  {folder.name}: {kept} tiles guardados, {skipped} descartados por parecer suelo vacio")


def main():
    parser = argparse.ArgumentParser(description="Recorta fotos de dron en tiles para etiquetado.")
    parser.add_argument("--input", required=True, help="Carpeta con las fotos originales (o subcarpetas por vuelo)")
    parser.add_argument("--output", required=True, help="Carpeta donde se guardan los tiles")
    parser.add_argument("--tile-size", type=int, default=640, help="Tamano del tile en pixeles (default: 640)")
    parser.add_argument("--overlap", type=float, default=0.15, help="Traslape entre tiles, 0-0.5 (default: 0.15)")
    parser.add_argument("--min-pixels", type=int, default=150,
                         help="Minimo de pixeles distintos al fondo para conservar un tile (default: 150). "
                              "Sube este valor si quedan tiles de puro suelo vacio; bajalo si se estan "
                              "perdiendo tiles con plantas pequenas y muy dispersas (etapa de emergencia).")
    args = parser.parse_args()

    input_dir = Path(args.input)
    output_dir = Path(args.output)

    if not input_dir.exists():
        print(f"Error: no existe la carpeta de entrada {input_dir}", file=sys.stderr)
        sys.exit(1)

    output_dir.mkdir(parents=True, exist_ok=True)

    flights = find_flight_folders(input_dir)
    print(f"Vuelos/carpetas encontrados: {len(flights)}")

    manifest_rows = []
    for folder in flights:
        process_flight_folder(folder, output_dir, args.tile_size, args.overlap,
                               args.min_pixels, manifest_rows)

    manifest_path = output_dir / "manifest.csv"
    if manifest_rows:
        with open(manifest_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(manifest_rows[0].keys()))
            writer.writeheader()
            writer.writerows(manifest_rows)
        print(f"\nListo. {len(manifest_rows)} tiles generados en total.")
        print(f"Manifest guardado en: {manifest_path}")
        print("\nSiguiente paso: sube la carpeta de tiles (por vuelo) a Roboflow o CVAT para etiquetar.")
    else:
        print("\nNo se genero ningun tile. Revisa la carpeta de entrada y el umbral --min-pixels.")


if __name__ == "__main__":
    main()

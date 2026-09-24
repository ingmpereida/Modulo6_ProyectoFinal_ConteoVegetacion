# Conteo de Vegetación con IA — Documentación del Proyecto

Proyecto final del Módulo 6: pipeline de visión por computadora para **contar plantas en parcelas agrícolas** a partir de fotos de dron o celular. El objetivo del sistema es estimar la densidad de vegetación (especialmente en etapas de emergencia, donde las plantas son pequeñas y dispersas), primero con procesamiento clásico de color y, en fases posteriores, con un modelo entrenado.

Este README está escrito para que **una IA o una persona pueda entender el proyecto y continuar el trabajo sin re-descubrir nada**. Léelo completo antes de modificar código.

---

## Estado actual (resumen ejecutivo)

| Archivo | Rol | Estado |
|---|---|---|
| `conteo_vegetacion.html` | Fase 1 — prototipo de conteo clásico por color, 100 % en el navegador | Funcional, con worker |
| `tile_pipeline.py` | Fase 2 — preparación de imágenes para etiquetado (tiles + manifest) | Funcional |
| `requirements.txt` | Dependencias Python (`Pillow`, `numpy` + pipeline YOLO) | Creado |
| `tests/computeDetection.test.js` | Test automatizado de `computeDetection` (runner nativo de Node) | 8/8 pasando |
| `package.json` | Script `npm test` para correr los tests | — |
| `README.md` | Este documento | — |
| `prepare_dataset.py` | Fase 2–3 — build del dataset YOLO: split por vuelo + `data.yaml` | Hecho (PR2) |
| `train.py` | Fase 2–3 — entrenamiento YOLO26: train + val + métricas mAP | Hecho (PR3) |
| `infer.py` | Fase 3 — conteo con modelo entrenado: CLI determinista + CSV con dedup NMS (Módulo 4) | Hecho (cambio `yolo-inference`) |

El repositorio **git** se inicializó con el PR1 del cambio `yolo-training-pipeline` (rama `feat/yolo-pr1-repo`, base de la cadena `feat/yolo-training-pipeline`, commits convencionales); `.gitignore` excluye artefactos generados (`runs/`, `datasets/`, venv, caches, `node_modules/`). La lógica de detección sí tiene tests automatizados — ver *Tests automatizados*.

---

## Pipeline global (fases del proyecto)

```
Fase 1 (hecha)        Fase 2 (hecha)              Fase 2–3 (hecha)          Fase 3 (hecha)
Fotos de dron     ->  Cortar en tiles          ->  Dataset YOLO (por vuelo) -> Conteo con el modelo
conteo clásico        descartar tiles vacíos       entrenar YOLO26 + val     entrenado (infer.py,
por color             generar manifest.csv         mAP                      Módulo 4)
```

- **Fase 1**: `conteo_vegetacion.html` — conteo por color (Excess Green Index) + componentes conexas. Prototipo interactivo.
- **Fase 2**: `tile_pipeline.py` — convierte fotos grandes en tiles etiquetables, descartando suelo vacío.
- **Fase 2–3 (hecha)**: `prepare_dataset.py` convierte tiles + etiquetas YOLO en un dataset con split por vuelo; `train.py` entrena YOLO26 y reporta mAP50/mAP50-95 (ver Módulo 3).
- **Fase 3 (hecha)**: `infer.py` cuenta plantas en fotos con un modelo entrenado, deduplicando por IoU en coordenadas globales (ver Módulo 4). Pendiente: correr un entrenamiento real con datos etiquetados de las parcelas.

---

## Módulo 1 — `conteo_vegetacion.html` (Fase 1, prototipo)

Aplicación de **un solo archivo HTML** (CSS + JS embebidos), en español, sin dependencias externas (solo fuentes de Google Fonts). Se abre directamente desde el escritorio con doble clic — **no requiere servidor**. La imagen nunca sale del navegador.

### Flujo del usuario

1. Subir/drag&drop/tomar foto (JPG, PNG, TIF).
2. La app muestra un preview (máx. 1100×720) y calcula el conteo automáticamente.
3. Ajustar 3 controles de sensibilidad (al soltarlos, recalcula solo):
   - **Índice de verdor**: umbral de Excess Green Index (default 18, rango −40…80).
   - **Tamaño mínimo**: píxeles mínimos de una mancha para contarla (default 14 px, ahora en píxeles **de la imagen real**, no del preview).
   - **Separar grupos unidos**: toggle que estima varias plantas dentro de una mancha grande.
4. Los `readouts` muestran: plantas detectadas, manchas analizadas y % de cobertura verde.
5. **Descargar imagen marcada**: exporta la imagen **original completa** (no el preview) con los círculos dibujados.

### Algoritmo de detección

Vive en la función pura **`computeDetection(w, h, data, threshold, minSize, splitClusters)`**:

1. **Máscara binaria** por Excess Green Index: `ExG = 2G − R − B`; un píxel es "verde" si `ExG > threshold`.
2. **Componentes conexas** (4-conectividad) con flood fill **iterativo** (pilas `Int32Array`, no recursión — evita desbordar el call stack en imágenes grandes).
3. **Separación de grupos**: la mediana de las áreas de los componentes estima el área "típica" de una planta; los componentes con `area > max(mediana × 2.4, minSize × 3)` se dividen en `round(area / mediana)` sub-marcadores distribuidos en grilla sobre su bounding box.

Los marcadores se guardan en **coordenadas de la imagen original**; el dibujo los escala (`displayScale`) para el preview.

### Decisiones técnicas (y por qué)

| Decisión | Detalle |
|---|---|
| **Conteo a resolución completa** | El análisis corre sobre `workCanvas` (canvas offscreen con los píxeles reales), NO sobre el preview re-escalado. Antes contaba sobre 1100×720 y perdía plantas chicas. |
| **Web worker embebido como Blob URL** | La página no se congela durante el procesamiento (1–3 s en fotos de 20–40 MP). Se usa Blob URL y NO un archivo `worker.js` separado porque **Chrome bloquea workers desde `file://`**; el blob mantiene la app de un solo archivo. |
| **Una sola fuente de verdad** | El worker se construye serializando `computeDetection.toString()` dentro del blob. La lógica existe una sola vez; inline y worker siempre coinciden. |
| **Umbral `WORKER_MIN_PIXELS = 5 000 000` (~5 MP)** | Imágenes menores se procesan inline (el overhead de mensajería no vale la pena); mayores van al worker con transferencia del buffer **zero-copy** (`postMessage(msg, [buffer])`). |
| **`detectionSeq`** | Contador que invalida resultados viejos (recálculos rápidos o resultados en vuelo al hacer "Quitar"). |
| **Fallback automático** | Si el worker falla (`onerror`), se desactiva y se vuelve a procesar inline. El usuario nunca queda colgado. |
| **Descarga con `toBlob` + JPEG (0.95)** | `toDataURL('image/png')` genera una cadena base64 gigante que revienta la memoria en fotos grandes y falla en silencio. `toBlob` + `URL.createObjectURL` es el patrón robusto; JPEG pesa ~5–10 % del PNG. Fallback PNG para navegadores viejos. |
| **Evento `change` (no `input`) en sliders** | Recalcula al soltar el control, no en cada tick — evita disparar procesamiento en cascada inútil. |

### Limitaciones conocidas (Fase 1)

- Las fotos muy grandes consumen memoria: `getImageData` de 40 MP ≈ 160 MB de buffer + canvas. Aceptable para prototipo, ojo en máquinas débiles.
- La división de grupos es una **heurística** (grilla sobre el bbox): en manchas alargadas reparte marcadores fuera de la máscara real. Sirve para estimar, no para medir exacto.
- El flood fill bloquea el hilo principal en el camino inline (fotos < 5 MP) y ocupa memoria en ambos caminos; la mejora natural es `OffscreenCanvas` dentro del worker.
- El conteo es por color: sufre con sombras, suelo con residuos vegetales, luz cambiante.

---

## Módulo 2 — `tile_pipeline.py` (Fase 2, preparación de tiles)

Script Python que corta fotos de dron en tiles pequeños (para que cada planta ocupe suficientes píxeles) y **descarta tiles "vacíos"** (suelo uniforme sin nada que etiquetar), generando un `manifest.csv` para trazar cada tile hasta su foto original.

### Uso

```bash
python3 tile_pipeline.py --input ./fotos_dron --output ./tiles

# Parámetros ajustados
python3 tile_pipeline.py --input ./fotos_dron --output ./tiles \
    --tile-size 768 --overlap 0.15 --min-pixels 150
```

### Parámetros CLI

| Parámetro | Default | Descripción |
|---|---|---|
| `--input` | (requerido) | Carpeta con fotos o subcarpetas por vuelo |
| `--output` | (requerido) | Carpeta de salida de tiles + manifest |
| `--tile-size` | 640 | Tamaño del tile en píxeles |
| `--overlap` | 0.15 | Traslape entre tiles (0–0.5) |
| `--min-pixels` | 150 | Mínimo de píxeles distintos del fondo para conservar un tile |

Requiere: `pip install -r requirements.txt`

### Estructura de entrada/salida

```
input_dir/
  ParcelaA_2026-09-10_emergencia/   ← cada subcarpeta = un vuelo
    DJI_0001.JPG                      (parcela + fecha + etapa)
    DJI_0002.JPG
  ParcelaB_2026-09-12_emergencia/
    ...

output_dir/
  ParcelaA_2026-09-10_emergencia/
    DJI_0001_x00000_y00000.jpg      ← tile con naming posicional
    DJI_0001_x00768_y00000.jpg
  manifest.csv
```

Reglas de comportamiento:
- Si `input_dir` no tiene subcarpetas, se trata como un único vuelo.
- Imágenes más chicas que el tile se copian enteras (posición `(0,0)`, tamaño real).
- El **traslape es aproximado**: el stride se calcula como `int(tile_size × (1 − overlap))`; el último tile de cada fila/columna se empuja hacia adentro para no salirse del borde.

### Esquema del `manifest.csv`

| Columna | Significado |
|---|---|
| `vuelo` | Nombre de la subcarpeta (parcela_fecha_etapa) |
| `imagen_origen` | Foto original (ej. `DJI_0001.JPG`) |
| `tile` | Nombre del tile generado |
| `x_offset`, `y_offset` | Coordenadas de la esquina superior-izquierda en la foto original |
| `tile_w`, `tile_h` | Tamaño real del tile (puede ser menor en bordes/imágenes chicas) |
| `imagen_ancho`, `imagen_alto` | Dimensiones de la foto original |

El manifest permite **auditar o reconstruir** el conteo por foto/parcela si se etiqueta a nivel de tile.

### Algoritmo de descarte de tiles vacíos — `has_enough_detail()`

En vez de medir la varianza total del tile (baja cuando las plantas cubren poca área, típico en emergencia), cuenta cuántos píxeles se distinguen del color de fondo dominante:

```
fondo = mediana de todos los píxeles RGB del tile
diff  = |píxel − fondo| sumado sobre los 3 canales
tile conservado si (diff > 25).count() >= min_pixels
```

`min_pixels` alto = más estricto (descarta más tiles casi vacíos); bajo = conserva tiles con plantas muy dispersas.

### Problemas conocidos

1. **Duplicación de plantas entre tiles:** el traslape hace que una misma planta aparezca en tiles adyacentes. Está bien para etiquetar (no se cortan copas), pero si el conteo se hace sumando etiquetas por tile hay que **deduplicar** usando las coordenadas del manifest.
2. No hay `--dry-run` ni estadísticas de memoria; para datasets grandes conviene ir por vuelos.

---

## Módulo 3 — `prepare_dataset.py` y `train.py` (Fase 2–3, entrenamiento YOLO)

Dos scripts convierten los tiles + etiquetas de la Fase 2 en un dataset YOLO y entrenan un detector **YOLO26 nano** (`yolo26n.pt`) para contar plantas. Requieren **Python 3.14** y `pip install -r requirements.txt` (ultralytics arrastra torch/torchvision automáticamente).

### 3.1 `prepare_dataset.py` — construir el dataset

```bash
python3 prepare_dataset.py --manifest tiles/manifest.csv --labels tiles --output datasets/plantas
```

| Parámetro | Default | Descripción |
|---|---|---|
| `--manifest` | (requerido) | `manifest.csv` generado por `tile_pipeline.py` |
| `--labels` | (requerido) | Raíz con carpetas por vuelo (imágenes + sidecars `.txt`) |
| `--output` | (requerido) | Carpeta del dataset (se sobrescribe si ya existe) |
| `--valid-ratio` | 0.2 | Fracción de validación por vuelo, en (0, 1) |
| `--seed` | 42 | Semilla del split (mismos inputs + seed ⇒ mismo layout) |

Códigos de salida: `0` éxito, `1` error de datos (**no se escribe nada**), `2` uso inválido.

### 3.2 Estructura del dataset (salida)

```
datasets/plantas/
  images/train/   labels/train/    ← tiles + sidecars de entrenamiento
  images/valid/   labels/valid/    ← tiles + sidecars de validación
  data.yaml                        ← path, train, val, names: [plant]
```

### 3.3 Split por vuelo — racionalidad y limitación conocida

El split train/valid es **por vuelo** (80/20, nunca se cruzan tiles entre vuelos): los tiles de un mismo vuelo se traslapan (el overlap de `tile_pipeline.py`) y comparten plantas, así que un split global pondría vistas de la misma planta en train y valid al mismo tiempo (leakage) e inflaría las métricas. Regla de borde: un vuelo con 1 solo tile va completo a train y deja valid vacío.

**Limitación (leakage intra-vuelo, aceptada):** dentro de un mismo vuelo, dos tiles solapados pueden caer uno en train y otro en valid, es decir, la misma planta puede verse en ambos conjuntos. Las métricas del modelo son por lo tanto **indicativas**, no una evaluación rigurosa de generalización entre vuelos; para evaluar de verdad, separar vuelos completos (train con vuelos A/B, valid con vuelo C).

### 3.4 Etiquetas: sidecar faltante y archivos malformados

Las etiquetas son sidecars YOLO `<tile>.txt` al lado de la imagen (formato `class x y w h`, normalizado, una caja por línea).

- **Tile sin sidecar** = muestra de fondo: la imagen se copia solo a `images/…`, **nunca** a `labels/…` (`labels/` solo contiene sidecars válidos). Un sidecar vacío es una etiqueta válida "sin objetos", distinta del fondo.
- **Sidecar malformado** (línea que no tiene 5 campos, clase no entera o coordenada no flotante) = **error duro**: el script falla con un mensaje claro y **no escribe nada** — el fallo es a propósito (fail fast), para que un dataset corrupto jamás se exporte en silencio.

### 3.5 `train.py` — entrenar y validar

```bash
python3 train.py --data datasets/plantas/data.yaml --epochs 100 --imgsz 640 --batch 8
# continuar un entrenamiento desde su último checkpoint:
python3 train.py --data datasets/plantas/data.yaml --resume runs/train/exp/weights/last.pt
```

| Parámetro | Default | Descripción |
|---|---|---|
| `--data` | (requerido) | `data.yaml` generado por `prepare_dataset.py` |
| `--weights` | `yolo26n.pt` | Pesos preentrenados (`.pt`) o un `.yaml` de arquitectura |
| `--epochs` | 100 | Épocas de entrenamiento |
| `--imgsz` | 640 | Tamaño de imagen de entrenamiento |
| `--batch` | 8 | Tamaño de lote (bajo para CPU) |
| `--project` | `runs/` | Raíz de artefactos de entrenamiento (gitignored) |
| `--resume` | (ninguno) | Ruta a un `last.pt` para continuar |

Flujo: carga el modelo, entrena (`model.train(...)`), valida (`model.val()`) e imprime `Validation metrics - mAP50 (B): x | mAP50-95 (B): y`. Pesos y métricas quedan bajo `runs/` (ignorado por git). Sin GPU, ultralytics usa CPU automáticamente (NFR-4); el primer uso descarga `yolo26n.pt`. Con `--resume`, los hiperparámetros guardados en el checkpoint tienen prioridad. La clase `0` corresponde a `plant`.

---

## Módulo 4 — `infer.py` (Fase 3, conteo con modelo entrenado)

CLI que cuenta plantas en fotos de dron usando el modelo entrenado en el Módulo 3 y los tiles + `manifest.csv` de la Fase 2 (Módulo 2). Es **determinista** (NFR-3): mismas entradas y flags ⇒ mismo CSV byte a byte — sin timestamps, rutas absolutas ni RNG en la salida.

> ⚠️ **Qué es y qué no es**: el motor de conteo con IA es **una herramienta CLI de Python**, NO IA dentro del navegador. El HTML `conteo_vegetacion.html` conserva su conteo clásico por color; `infer.py` es la vía con modelo entrenado, separada de la app web (non-goal: integración ONNX/Web fuera de alcance).

### Uso

```bash
# Modo directorio (salida de tile_pipeline.py: manifest.csv + tiles/)
python infer.py --weights runs/train/exp/weights/best.pt \
    --input ./input_tiles --output counts.csv

# Modo foto única (sin manifest, deduplicación desactivada)
python infer.py --weights runs/train/exp/weights/best.pt \
    --input DJI_0001.JPG --output counts.csv

# Con resumen por vuelo y progreso detallado
python infer.py --weights runs/train/exp/weights/best.pt \
    --input ./input_tiles --output counts.csv --summary --verbose
```

### Parámetros CLI

| Parámetro | Default | Descripción |
|---|---|---|
| `--weights` | (requerido) | Pesos entrenados `.pt` (ej. `runs/train/exp/weights/best.pt`). Debe existir y ser un archivo legible; si no → código 2 |
| `--input` | (requerido) | Carpeta con `manifest.csv` + `tiles/` (layout de tile_pipeline.py) **o** una foto JPG/PNG/TIF suelta |
| `--output` | (requerido) | Ruta del CSV de salida. Se escribe SOLO si todas las fotos se procesan bien (NFR-6) |
| `--conf` | 0.25 | Confianza mínima de detección, en el intervalo abierto (0, 1) |
| `--iou` | 0.5 | Umbral IoU del NMS global, en el intervalo abierto (0, 1) |
| `--summary` | off | Imprime totales por vuelo en stdout; NO altera el CSV |
| `--verbose` | off | Imprime una línea de progreso por foto |

### Layout de entrada (modo directorio)

```
input_tiles/
  manifest.csv              ← generado por tile_pipeline.py (Módulo 2)
  tiles/
    ParcelaA_2026-09-10_emergencia/
      DJI_0001_x00000_y00000.jpg
      ...
```

Modo foto única: se pasa directamente el archivo de imagen (sin manifest). Ahí no hay offsets contra los cuales deduplicar, así que el conteo omite el NMS (`dedup_removed = 0`, `box_count = global_count`) y la fila del CSV usa `flight = photo = nombre base` (D6).

### Códigos de salida

| Código | Significado |
|---|---|
| `0` | Éxito: CSV escrito |
| `1` | Error de datos/procesamiento — **nada se escribe**: falta `manifest.csv`, un tile referenciado no existe, pesos corruptos pero legibles, fallo a mitad de corrida |
| `2` | Uso inválido (argparse): faltan argumentos, `--weights` inexistente/no legible, `--conf`/`--iou` fuera de (0, 1), `--input` inexistente |

### Por qué la deduplicación (y qué hace el NMS)

El traslape entre tiles (Módulo 2) hace que una misma planta aparezca en tiles adyacentes. `infer.py` no suma detecciones por tile como si fueran plantas distintas: proyecta cada detección a **coordenadas globales de la foto** (desplazando por `x_offset`/`y_offset` del manifest), recorta las que quedan parcialmente fuera de la foto, y luego aplica **IoU NMS por foto**:

- Se ordenan las cajas por confianza descendente (empates: orden de aparición — determinista, sin RNG).
- Se conserva una caja si su IoU con TODAS las ya conservadas es `< --iou` (default 0.5); si IoU `≥ 0.5` se considera la misma planta y se descarta.
- `global_count` = cajas conservadas; `box_count` = detecciones que pasaron el filtro de clase+confianza; `dedup_removed = box_count − global_count`.
- `source_tiles` = lista ordenada de los tiles que aportaron ≥ 1 caja conservada (semantic: los dos tiles de una planta duplicada aparecen listados, FR-4).

El CSV usa el header exacto `flight,photo,global_count,box_count,dedup_removed,source_tiles`, una fila por foto, ordenadas por `(flight, photo)`.

---

## Flujo de datos completo (cómo encajan las piezas)

```
Foto de dron (celular/dron/ortomosaico)
  │
  ├─► Fase 1: conteo_vegetacion.html (humano, en el navegador)
  │     plan → preview → ajustar sensibilidad → imagen original marcada + conteo
  │
  └─► Fase 2: tile_pipeline.py (preparación para ML)
        fotos → tiles etiquetables + manifest.csv
        → etiquetas YOLO (sidecars .txt) → prepare_dataset.py → train.py
        → modelo entrenado (mAP en consola) → infer.py (Módulo 4)
        → counts.csv (conteo por foto y por vuelo)

> Nota de alcance: el conteo con IA (`infer.py`) es un CLI de Python; la app
> HTML `conteo_vegetacion.html` sigue con su camino clásico por color (Fase 1).
```

---

## Convenciones del proyecto

| Ámbito | Convención |
|---|---|
| Idioma del código (identificadores) | Inglés |
| Idioma de UI/copy de cara al usuario | Español (neutro, sin regionalismos) |
| Comentarios en código | Mixto histórico (ES en `tile_pipeline.py`, EN en las secciones nuevas del HTML). Para código nuevo: inglés. |
| Nombres de vuelo | `ParcelaX_YYYY-MM-DD_etapa` (ej. `ParcelaA_2026-09-10_emergencia`) |
| Nombres de tile | `<foto_original>_x<offset_X 5 dígitos>_y<offset_Y 5 dígitos>.jpg` |
| Commits | Conventional Commits (repo inicializado en el PR1 de `yolo-training-pipeline`; respetar el historial existente) |

---

## Cómo verificar el proyecto

### Frontend (`conteo_vegetacion.html`)

1. Abrir el archivo con doble clic en un navegador moderno (Chrome/Firefox/Edge actual; requiere `Worker`, canvas y ES2021+).
2. Subir una foto con vegetación → verificar conteo, manchas y % de cobertura.
3. Mover sliders y soltar → debe recalcular sin congelar la página (camino worker para imágenes grandes).
4. Descargar → verificar que la imagen resultante tiene las dimensiones originales y los marcadores.

### Verificación de sintaxis del JS embebido (Windows/PowerShell + node)

```powershell
$html = Get-Content -Raw -LiteralPath "conteo_vegetacion.html"
$m = [regex]::Match($html, '<script>\s*(.*?)\s*</script>', 'Singleline')
Set-Content -LiteralPath "$env:TEMP\conteo_check.js" -Value $m.Groups[1].Value -Encoding utf8
node --check "$env:TEMP\conteo_check.js"
```

### Tests automatizados (Node)

La lógica de detección (`computeDetection`) tiene tests con el runner **nativo de Node** (`node:test`), sin dependencias externas. Los tests extraen la función del HTML contando llaves (la misma técnica que usa la app para serializarla en el worker) y la prueban con imágenes sintéticas.

```bash
npm test      # equivalente a: node --test
```

El archivo `tests/computeDetection.test.js` cubre: blob aislado, filtro `minSize`, dos blobs separados, blob pegado al borde, imagen sin verde, `splitClusters` (28 marcadores para la imagen de prueba) y determinismo de la mediana.

> Sugerencia para una IA futura: si `computeDetection` deja de pasar estos tests, el problema está en la lógica compartida del conteo — no dupliques la lógica en el worker para "arreglar" el test.

### Backend (`tile_pipeline.py`)

```bash
# Crear estructura mínima de prueba
mkdir -p fotos_dron/ParcelaX_2026-09-10_emergencia
# copiar una foto JPG cualquiera dentro
python3 tile_pipeline.py --input ./fotos_dron --output ./tiles
# esperado: tiles generados + manifest.csv con columnas correctas
```

---

## Roadmap y próximos pasos sugeridos

1. ~~Inicializar **git** (repo + commits conventionales) y añadir `.gitignore`.~~ — **hecho** en el PR1 del cambio `yolo-training-pipeline`.
2. **Fase 2–3** (cambio `yolo-training-pipeline`): ~~etiquetar tiles en Roboflow/CVAT usando el `manifest.csv`; `prepare_dataset.py` con split por vuelo y `data.yaml` (PR2) y `train.py` con YOLO26 + validación (PR3)~~ — **hecho**: ver Módulo 3; queda pendiente etiquetar datos reales y correr el primer entrenamiento.
3. **Fase 3** (cambio `yolo-inference`): ~~`infer.py` — CLI determinista de conteo con modelo entrenado, dedup por IoU global, modos directorio/foto única, exit codes 0/1/2~~ — **hecho**: ver Módulo 4. Una vez existan pesos reales (punto 2), correr `python infer.py --weights runs/train/exp/weights/best.pt --input <carpeta con manifest.csv + tiles/> --output counts.csv`.
4. Optimización opcional: `OffscreenCanvas` dentro del worker para evitar el `getImageData`/transferencia en el hilo principal.

---

## Registro de cambios recientes (para contexto de una IA futura)

| Cambio | Qué se hizo y por qué |
|---|---|
| **Conteo a resolución completa** | El análisis pasó de correr sobre el preview (1100×720) a correr sobre los píxeles originales vía `workCanvas` offscreen; los marcadores se almacenan en coordenadas full-res y se escalan solo para dibujar. La descarga pasó a exportar la imagen original marcada. |
| **Web worker embebido** | Extraída la lógica a la función pura `computeDetection`; worker creado como Blob URL serializando la función (`toString()`); umbral de 5 MP para decidir worker vs inline; guarda `detectionSeq` contra resultados viejos; fallback inline en `onerror`; sliders recalculan al soltar (`change`). |
| **Descarga robusta** | `toDataURL` PNG → `toBlob` JPEG 0.95 + `URL.createObjectURL` (+ revoke), para que fotos de 20–40 MP descarguen la original con marcas sin reventar la memoria. |
| **Fix fondo negro al descargar** | `drawMarkers` hacía `clearRect` al inicio, lo que borraba la foto recién dibujada en el canvas de exportación (el JPEG codificaba la transparencia como negro). Ahora `drawMarkers(..., clear = true)` y el export usa `clear=false`. |
| **Fix `--min-detail` + tests automatizados** | Corregida la referencia al parámetro en el mensaje final de `tile_pipeline.py`. Nuevo `tests/computeDetection.test.js` (8 casos sobre el runner nativo de Node, `npm test`) y `package.json` mínimo. |
| **UI/UX v2** | Feedback de procesamiento (botón con spinner y badge "analizando a resolución completa…" con anillo), metadatos de la imagen cargada (nombre + dimensiones, deja claro que se procesa la original), guía contextual cuando no se detectan plantas (sugiere bajar Índice de verdor o Tamaño mínimo), toasts con `role="status"` (descarga exitosa, archivo inválido, fallback sin worker), accesibilidad (`aria-live="polite"` en los readouts), micro-interacciones (pulse del contador, hover con sombra en la zona de subida) y ajustes para pantallas ≤ 480 px. |
| **CSS profesional (auditoría de UI/UX)** | Cierre de huecos detectados en revisión del CSS: slider estilizado para Firefox (`::-moz-range-track/thumb`), todos los colores pasaron a tokens (`--sand`, `--sand-tint`, `--switch-off`, `--sage-bright`, `--warn-*`, `--danger` ya existía; queda literal solo `#fff`), `prefers-reduced-motion` para desactivar animaciones, `color-scheme: light` (evita estilos oscuros de UA en controles nativos), `-webkit-tap-highlight-color` transparente en móvil, `aria-hidden="true"` en las 4 SVGs decorativas (incluida la del template de `buildStage`), selector `.readout` duplicado unificado, y en ≤860 px el resultado pasa primero (`order:-1`) con header apilado. Verificado: llaves CSS balanceadas, `node --check` OK, 8/8 tests. |
| **Módulo 4 — `infer.py`** (cambio `yolo-inference`) | Motor de conteo con modelo entrenado como CLI de Python: `load_model` es el único seam a ultralytics (import diferido dentro de la función, NFR-1) y el resto de la pipeline consume el duck-type `predict(image) -> [Box]` — todo corre en CPU sin runtime. `main()`/argparse (FR-1): `--weights` (existente y legible, exit 2 si no), `--input` (directorio con `manifest.csv` + `tiles/` o foto única), `--output`, `--conf` 0.25, `--iou` 0.5, `--summary`, `--verbose`; salidas 0/1/2, nada se escribe ante error (NFR-6, validate-then-write). Dedup por proyección global + IoU NMS determinista (sin RNG); CSV byte-idéntico entre corridas (NFR-3). Tests: 8 subprocess con paquete `ultralytics` falso en PYTHONPATH + unit del adapter. No es IA en el navegador: la app HTML mantiene su conteo clásico. |

## Relación con la memoria persistente (Engram)

Si esta sesión o una futura tiene acceso al servidor MCP de **Engram** (proyecto `proyectofinal-modulo6`), las decisiones no obvias también están guardadas como observaciones (`discovery/`, `bug/…`, `architecture/web-worker…`). En caso de duda entre este README y una observación, **el código y este README mandan**; Engram es contexto complementario.
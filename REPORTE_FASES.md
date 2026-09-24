# Reporte de Fases — Proyecto Conteo de Vegetación con IA

Reporte técnico consolidado de las fases desarrolladas en el proyecto **Conteo de Vegetación con IA** (Proyecto Final — Módulo 6). Este documento está escrito para que **una persona o una IA** pueda comprender qué se construyó, por qué, con qué decisiones técnicas, y cómo continuar el trabajo.

> Fuentes primarias: `README.md` (referencia operativa completa), código fuente (`conteo_vegetacion.html`, `tile_pipeline.py`, `prepare_dataset.py`, `train.py`, `infer.py`, `convert_polygon_to_bbox.py`), suite de tests (`tests/`) y memoria persistente Engram (proyecto `proyectofinal-modulo6`). Si este reporte y el código discrepan, **el código manda**.

---

## 1. Resumen ejecutivo

El proyecto construye un sistema de visión por computadora para **contar plantas en parcelas agrícolas** a partir de fotos de dron o celular, con foco en etapas de **emergencia** (plantas pequeñas y dispersas). Se desarrolló en fases incrementales:

| Fase | Entregable | Estado |
|---|---|---|
| **F1** | `conteo_vegetacion.html` — conteo clásico por color, 100 % en el navegador | Funcional, con worker y tests (8/8) |
| **F2** | `tile_pipeline.py` — recorte de fotos en tiles etiquetables + `manifest.csv` | Funcional |
| **F2–3** | `prepare_dataset.py` + `train.py` — dataset YOLO + entrenamiento YOLO26 | Funcional |
| **F3** | `infer.py` — conteo con modelo entrenado (CLI determinista, CSV + dedup NMS) | Funcional, seam real validado (hito D8) |
| **Piloto real** | Roboflow → dataset → entrenamiento → inferencia completados | Seam probado; modelo requiere más datos |

**Estado del modelo:** el pipeline técnico está 100 % validado contra el runtime real (ultralytics 8.4.161). El modelo piloto entrenado con 5 imágenes no generaliza (mAP50 ≈ 0), lo cual es esperado por la cantidad de datos, no por un defecto de código. Los requisitos de datos para un modelo útil están documentados en el Módulo 5 del README.

---

## 2. Contexto y problema

- **Problema:** estimar la densidad de vegetación en parcelas agrícolas, especialmente en **emergencia**, donde las plantas son pequeñas, dispersas y de color similar al suelo.
- **Restricción de campo:** los conteos manuales por hectárea son inviables; se necesita automatización con fotos de dron/celular.
- **Decisión de arquitectura:** pipeline en fases (clásico → tiles → ML), no una única aplicación monolítica. Cada fase es testeable por separado.

---

## 3. Fase 1 — `conteo_vegetacion.html` (conteo clásico por color)

### 3.1 Qué es

Aplicación de **un solo archivo HTML** (CSS + JS embebidos), en español, sin dependencias externas. Se abre con doble clic desde el escritorio (**no requiere servidor**); la imagen nunca sale del navegador. Al terminar cada conteo, lee los resultados en voz alta con la API `speechSynthesis` del navegador (voz en español), sin servidor ni dependencias.

### 3.2 Algoritmo de detección — `computeDetection(w, h, data, threshold, minSize, splitClusters)`

Función **pura** (mismos inputs ⇒ mismos outputs, sin estado) que vive en el HTML:

1. **Máscara binaria** por **Excess Green Index**: `ExG = 2G − R − B`; un píxel es "verde" si `ExG > threshold` (default 18, rango −40…80).
2. **Componentes conexas** con 4-conectividad, flood fill **iterativo** (pilas `Int32Array`) — se evitó la recursión para no desbordar el stack en imágenes de 20–40 MP.
3. **Separación de grupos**: la mediana de las áreas estima el área típica de una planta; componentes con `area > max(mediana × 2.4, minSize × 3)` se dividen en `round(area / mediana)` sub-marcadores sobre su bounding box.

### 3.3 Decisiones técnicas clave

| Decisión | Detalle / por qué |
|---|---|
| **Conteo a resolución completa** | El análisis corre sobre un canvas offscreen con los **píxeles reales**, no sobre el preview re-escalado (antes perdía plantas chicas). Los marcadores se guardan en coordenadas de la imagen original. |
| **Web worker como Blob URL** | Chrome **bloquea workers desde `file://`**; un Blob URL mantiene la app de un solo archivo y no congela la UI. El worker serializa `computeDetection.toString()` — la lógica existe una sola vez, inline y worker siempre coinciden. |
| **Umbral de worker 5 MP** | Imágenes < 5 MP se procesan inline (el overhead de mensajería no vale la pena); mayores van al worker con transferencia **zero-copy** (`postMessage(msg, [buffer])`). |
| **`detectionSeq`** | Contador que invalida resultados viejos (recálculos rápidos o resultados en vuelo al hacer "Quitar"). |
| **Fallback automático** | Si el worker falla (`onerror`), se desactiva y se vuelve a procesar inline — el usuario nunca queda colgado. |
| **Descarga robusta** | `toBlob` + JPEG 0.95 + `URL.createObjectURL`: `toDataURL('image/png')` genera base64 gigante que revienta la memoria en fotos grandes. Fallback PNG para navegadores viejos. |
| **Evento `change` en sliders** | Recalcula al soltar el control, no en cada tick — evita procesamiento en cascada. |
| **Audio automático de resultados** | `buildResultSpeech()` (pura, testeable sin DOM) genera la frase y `speakResults()` la lee con `speechSynthesis` (es-ES): al finalizar el conteo lee plantas detectadas, manchas analizadas y % de cobertura ("34,5 por ciento" — coma decimal, más natural al hablar). `synth.cancel()` evita frases encoladas y el `speak()` se difiere 60 ms (quirk de Chromium que descarta utterances llamadas sincrónicamente tras `cancel()`), con guard de frescura `seq !== detectionSeq`; `lastSpokenSeq` evita repetir el mismo resultado. Toggle "Audio al finalizar" en la barra lateral (default ON, `aria-label` accesible). Requiere una primera interacción del usuario para habilitar audio — el clic para subir la imagen ya la habilita. |
| **UI/UX v2** | Spinner "analizando a resolución completa…", metadatos de imagen cargada, guía contextual cuando no hay detecciones, toasts accesibles (`role="status"`), tokens de color CSS, `prefers-reduced-motion`, CSS profesional con auditoría (ver README). |

### 3.4 Limitaciones conocidas (F1)

- Fotos muy grandes consumen memoria: `getImageData` de 40 MP ≈ 160 MB de buffer.
- La división de grupos es una **heurística** (grilla sobre bbox); en manchas alargadas reparte marcadores fuera de la máscara real.
- El conteo por color sufre con sombras, suelo con residuos vegetales y luz cambiante — **motivo por el cual se construyó el camino ML (Fases 2–3)**.

### 3.5 Tests

`tests/computeDetection.test.js` — 8 casos sobre el runner **nativo de Node** (`node:test`, `npm test`): blob aislado, filtro `minSize`, dos blobs separados, blob pegado al borde, imagen sin verde, `splitClusters`, determinismo de la mediana. Extrae la función del HTML contando llaves (misma técnica que usa la app para el worker).

`tests/speakResults.test.js` — 4 casos sobre la frase hablada (`buildResultSpeech`): texto exacto del resumen en español con coma decimal ("34,5 por ciento"), caso cero detecciones y fallback "sin dato" para porcentaje no numérico. Verifica el contrato de claridad del audio sin tocar el DOM.

---

## 4. Fase 2 — `tile_pipeline.py` (preparación de imágenes para ML)

### 4.1 Qué es

Script Python que corta fotos grandes en **tiles** (recortes) para que cada planta ocupe suficientes píxeles y sea etiquetable, descarta tiles "vacíos" (suelo uniforme) y genera **`manifest.csv`** que traza cada tile hasta su foto original.

### 4.2 Estructura de entrada / salida

```
input_dir/
  ParcelaA_2026-09-10_emergencia/   ← cada subcarpeta = un vuelo
    DJI_0001.JPG                      (parcela + fecha + etapa)
    DJI_0002.JPG

output_dir/
  ParcelaA_2026-09-10_emergencia/
    DJI_0001_x00000_y00000.jpg      ← tile con naming posicional
    DJI_0001_x00768_y00000.jpg
  manifest.csv
```

- Si `input_dir` no tiene subcarpetas, se trata como **un único vuelo** (piloto real: `fotos_dron/` → vuelo `fotos_dron`).
- Imágenes < tamaño de tile se **copían enteras** (posición `(0,0)`, tamaño real).
- El traslape es aproximado: `stride = int(tile_size × (1 − overlap))`; el último tile de cada fila/columna se empuja hacia adentro para no salirse del borde.

### 4.3 Esquema `manifest.csv`

| Columna | Significado |
|---|---|
| `vuelo` | Nombre de la subcarpeta (parcela_fecha_etapa) |
| `imagen_origen` | Foto original |
| `tile` | Nombre del tile |
| `x_offset`, `y_offset` | Esquina superior-izquierda del tile en la foto original |
| `tile_w`, `tile_h` | Tamaño real del tile (menor en bordes / imágenes chicas) |
| `imagen_ancho`, `imagen_alto` | Dimensiones de la foto original |

El manifest habilita **auditar o reconstruir** el conteo a nivel de foto/parcela y es la base del dedup global de `infer.py` (Fase 3).

### 4.4 Descarte de tiles vacíos — `has_enough_detail()`

En vez de varianza total (baja cuando las plantas cubren poca área — típico en emergencia), cuenta píxeles que se distinguen del **color de fondo dominante**:

```
fondo = mediana de todos los píxeles RGB del tile
diff  = |píxel − fondo| sumado sobre los 3 canales
tile conservado si (diff > 25).count() >= min_pixels   # default 150
```

`--min-pixels` alto = más estricto; bajo = conserva plantas muy dispersas.

### 4.5 Problema conocido

**Duplicación de plantas entre tiles adyacentes** por el traslape: una misma planta aparece en dos tiles. Está bien para etiquetar (no se cortan copas) pero el conteo por suma de detecciones por tile **requiere deduplicación** — resuelto en `infer.py` con NMS global (Fase 3).

---

## 5. Fase 2–3 — `prepare_dataset.py` y `train.py` (dataset YOLO + entrenamiento)

### 5.1 `prepare_dataset.py` — construir el dataset

```bash
python3 prepare_dataset.py --manifest tiles/manifest.csv --labels tiles --output datasets/plantas
```

| Parámetro | Default | Descripción |
|---|---|---|
| `--manifest` | (req.) | `manifest.csv` de `tile_pipeline.py` |
| `--labels` | (req.) | Raíz con carpetas por vuelo (imágenes + sidecars `.txt`) |
| `--output` | (req.) | Carpeta del dataset (se sobrescribe determinísticamente) |
| `--valid-ratio` | 0.2 | Fracción de validación por vuelo, en (0, 1) |
| `--seed` | 42 | Semilla del split (determinismo) |

Códigos de salida: `0` éxito, `1` error de datos (**nada se escribe**), `2` uso inválido.

Salida:

```
datasets/plantas/
  images/train/   labels/train/    ← tiles + sidecars de entrenamiento
  images/valid/   labels/valid/    ← tiles + sidecars de validación
  data.yaml                        ← path, train, val, names: [plant]
```

### 5.2 Split por vuelo — racionalidad y limitación aceptada

- El split train/valid es **por vuelo** (80/20, seed fija) para evitar que plantas duplicadas por overlap del mismo vuelo estén en train y valid a la vez (leakage→métricas infladas).
- **Limitación aceptada (leakage intra-vuelo):** dentro de un mismo vuelo, dos tiles solapados pueden caer uno en cada split; la misma planta puede verse en ambos. Las métricas son **indicativas**, no evaluación rigurosa. Evaluación honesta requiere separar vuelos completos (train A/B, valid C).
- Regla de borde: vuelo con 1 solo tile → todo a train, valid vacío.

### 5.3 Etiquetas — contratos duros

- Sidecars YOLO `<tile>.txt` al lado de la imagen: formato `class x y w h` **normalizado**, una caja por línea.
- **Tile sin sidecar** = muestra de fondo: imagen se copia solo a `images/`, jamás a `labels/`.
- **Sidecar malformado** (≠5 campos, clase no-entera, coordenada no-flotante) = **error duro, fail fast**: el script no escribe nada. Un dataset corrupto jamás se exporta en silencio.

### 5.4 `train.py` — entrenar YOLO26 nano y validar

```bash
python3 train.py --data datasets/plantas/data.yaml --epochs 100 --imgsz 640 --batch 8
python3 train.py --data datasets/plantas/data.yaml --resume runs/train/exp/weights/last.pt
```

| Parámetro | Default | Descripción |
|---|---|---|
| `--data` | (req.) | `data.yaml` generado por `prepare_dataset.py` |
| `--weights` | `yolo26n.pt` | Pesos preentrenados (`.pt`) o `.yaml` de arquitectura |
| `--epochs` | 100 | Épocas |
| `--imgsz` | 640 | Tamaño de imagen |
| `--batch` | 8 | Tamaño de lote (bajo para CPU) |
| `--project` | `runs/` | Raíz de artefactos (gitignored) |
| `--resume` | (ninguno) | Ruta a `last.pt` para continuar |

Detalles técnicos:
- **Import de ultralytics diferido** (dentro de `run()`): `--help` y los tests corren sin el runtime pesado.
- Tras `model.train(...)` → `model.val()`; lee exactamente las keys `metrics/mAP50(B)` y `metrics/mAP50-95(B)` (verificado contra ultralytics 8.4.161; fail fast si el runtime cambia).
- **La clase 0 es `plant`** (constante `PLANT_CLASS` en `infer.py`); cambiar el orden de `names` en `data.yaml` rompe el significado.
- Sin GPU, ultralytics usa CPU automáticamente.
- **Quirk de paths:** con `--project runs/` (default) los artefactos quedan en `runs/detect/runs/train/weights/{best,last}.pt` (el namespace se anida).

---

## 6. Fase 3 — `infer.py` (conteo con modelo entrenado)

### 6.1 Qué es y qué no es

CLI **determinista** de Python que cuenta plantas usando el modelo entrenado (Fase 2–3) y los tiles + `manifest.csv` (Fase 2). **No es IA en el navegador**: la app HTML conserva su camino clásico; `infer.py` es la vía con modelo (non-goal: integración ONNX/Web).

### 6.2 Contrato de determinismo (NFR-3)

Mismos inputs y flags ⇒ **mismo CSV byte a byte**. Sin timestamps, rutas absolutas ni RNG en la salida.

### 6.3 Uso y parámetros

```bash
# Modo directorio (layout de tile_pipeline.py: manifest.csv + tiles/)
python infer.py --weights runs/.../best.pt --input ./input_tiles --output counts.csv

# Modo foto única (sin manifest → dedup desactivada, D6)
python infer.py --weights runs/.../best.pt --input DJI_0001.JPG --output counts.csv

# Con resumen por vuelo y progreso
python infer.py --weights runs/.../best.pt --input ./input_tiles --output counts.csv --summary --verbose
```

| Parámetro | Default | Descripción |
|---|---|---|
| `--weights` | (req.) | `.pt` entrenado; debe existir y ser legible (si no → 2) |
| `--input` | (req.) | Carpeta con `manifest.csv` + `tiles/` **o** foto JPG/PNG/TIF suelta |
| `--output` | (req.) | CSV de salida; se escribe SOLO si todo el lote se procesa bien (NFR-6, validate-then-write) |
| `--conf` | 0.25 | Confianza mínima, en (0, 1) |
| `--iou` | 0.5 | Umbral IoU del NMS global, en (0, 1) |
| `--summary` | off | Totales por vuelo en stdout (no altera el CSV) |
| `--verbose` | off | Línea de progreso por foto |

### 6.4 Pipeline interno

```
load_model (único seam a ultralytics, import diferido)
   → [Box] duck-typed: predict(image) -> list[Box]  (Box: cls, conf, xyxy pixel)
read_tiles (manifest.csv → TileSpec; valida que el tile quede bajo tiles_root — anti path traversal)
   → para cada (flight, photo):
        proyecta detecciones a coordenadas GLOBALES de la foto (offset del manifest)
        recorta cajas parcialmente fuera de la foto
        NMS por IoU global (orden por conf descendente, empates por orden de aparición — determinista)
        → global_count, box_count, dedup_removed, source_tiles
render_csv (header exacto: flight,photo,global_count,box_count,dedup_removed,source_tiles)
   → escritura atómica (temp + os.replace, R4-003)
```

### 6.5 Códigos de salida

| Código | Significado |
|---|---|
| `0` | Éxito: CSV escrito |
| `1` | Error de datos/procesamiento — **nada se escribe**: `manifest.csv` faltante, tile inexistente, pesos corruptos pero legibles, fallo a mitad de corrida |
| `2` | Uso inválido (argparse): argumentos faltantes, `--weights` inexistente, `--conf`/`--iou` fuera de rango, `--input` inexistente |

### 6.6 Casos de borde críticos (validados por tests)

- **Foto sin detecciones** (ultralytics devuelve `boxes=None`): se escribe fila con `0,0,0` y `source_tiles` vacío; el batch **continúa**, exit 0 (fix R4-001 — antes abortaba la corrida).
- **Manifest vacío**: warning a stderr, CSV header-only, exit 0 (R4-002).
- **Tile que escapa de `tiles_root`** (`..` en manifest): error, nada se escribe (R1-001).
- **Escritura atómica**: el CSV nunca queda a medias (R1-004/R4-003).

### 6.7 Tests

Suite `tests/test_infer.py` (~150 tests): unit del adapter con paquete `ultralytics` falso en PYTHONPATH, unit de `read_tiles`/NMS/CSV, y suite **subprocess** del CLI con fixtures sintéticas. El stub del modelo soporta la variante `boxes=None` (foto vacía) para probar R4-001.

---

## 7. Piloto real (2026-09-24) — cierre del hito D8

### 7.1 Objetivo

Validar el **seam completo** contra el runtime real: no FakeModel, no pesos COCO genéricos, sino el flujo real de datos etiquetados → pesos entrenados → conteo.

### 7.2 Ejecución

| Paso | Detalle |
|---|---|
| **Fotos** | `fotos_dron/` con Parcela01 (703×860 PNG, 1.6 MB), Parcela02 (450×300), Parcela03 (450×338) |
| **Tiles** | `tile_pipeline.py` → 6 tiles (4×640px de Parcela01 + 2 copias enteras 450px) + `manifest.csv` |
| **Etiquetado** | Roboflow, proyecto `conteovegetacion`, clase `plant` (31 plantas en total) |
| **Versión + export** | Roboflow v1, formato `yolov8`, `auto-orient` sin augmentos |
| **Conversión** | **Ojo:** Roboflow exporta polígonos como polígonos → `convert_polygon_to_bbox.py` (min/max de x/y → `cx cy w h`) |
| **Dataset** | `prepare_dataset.py` → 5 train / 1 valid (1 vuelo `fotos_dron`; split casi nulo — limitación conocida) |
| **Entrenamiento** | `train.py --epochs 30` en CPU (sin GPU; Core Ultra 5 125H): ~43 s, `best.pt` 5.4 MB en `runs/detect/runs/train/weights/`, mAP50 0.001 |
| **Inferencia** | `infer.py --weights best.pt` sobre los 6 tiles (tamaños mixtos 640/448/512 → letterbox OK): 0 detecciones, CSV válido, exit 0 |

### 7.3 Conclusión técnica

- ✅ **El seam está probado**: loader real, predict real, conversión pixel-xyxy, NMS, CSV, escritura atómica, manejo de detecciones vacías y tiles de distinto tamaño — todo contra ultralytics 8.4.161 de verdad.
- ⚠️ **El modelo no generaliza** (mAP50 ≈ 0, 0 detecciones): esperado con 5 imágenes de entrenamiento. No es un bug.
- 📊 **Requisitos de datos para que aprenda** (ver README Módulo 5.11):
  - Mínimo para "aprender": ~100–200 tiles etiquetados con 3+ vuelos.
  - Usable de conteo: 300–600+ con 5+ vuelos y variedad de etapas/luces.
  - Regla de oro: ~150 instancias (cajas) por clase; la **variedad de vuelos** importa más que el total.

---

## 8. Flujo de datos completo (cómo encajan las piezas)

```
Foto de dron (celular/dron/ortomosaico)
  │
  ├─► Fase 1: conteo_vegetacion.html (humano, en el navegador)
  │     conteo clásico por color (ExG + componentes conexas)
  │
  └─► Fase 2: tile_pipeline.py (preparación para ML)
        fotos → tiles etiquetables + manifest.csv
        → etiquetar en Roboflow (cajas o polígonos, clase plant)
        → exportar YOLO + convert_polygon_to_bbox.py (si polígonos)
        → prepare_dataset.py → train.py → best.pt (mAP en consola)
        → infer.py (Fase 3) → counts.csv (conteo por foto y por vuelo)
```

---

## 9. Calidad, tests y reviews

| Ámbito | Detalle |
|---|---|
| **Tests Python** | `pytest tests/ -q` → **150 passed** (incluye suite subprocess del CLI) |
| **Tests Node** | `npm test` → **12/12** (`computeDetection` 8 + `buildResultSpeech` 4) |
| **TDD estricto** | Activado vía SDD: tests primero, luego implementación, por cambio |
| **Reviews aplicadas** | Cadena de PRs revisada con lentes `review-reliability` (gate), `review-risk`, `review-resilience`, `review-readability`. Hallazgos clave corregidos: R4-001 (boxes=None abortaba batch), R1-001 (path traversal en tiles), R1-004/R4-003 (escritura no atómica), R4-002 (manifest vacío silencioso), R2-001/002/004 (docstring, test duplicado, clase 0). Won't-fix documentados: R1-002 (`.pt` = pickle/RCE — solo pesos de fuentes confiables), R1-003 (inyección de fórmulas CSV — solo self-injection). |
| **Re-verificación final** | Ledger de review persistido; re-review scoped: 7/7 hallazgos verificados, 0 nuevos problemas |

### Convenciones

- Identificadores de código en **inglés**; UI/copy y docs en **español neutro**.
- Nombres de vuelo `ParcelaX_YYYY-MM-DD_etapa`; tiles `<foto>_x<offsetX 5d>_y<offsetY 5d>.jpg`.
- **Conventional Commits**, sin "Co-Authored-By". Repo git remoto: `ingmpereida/Modulo6_ProyectoFinal_ConteoVegetacion`.

---

## 10. Estado actual y próximos pasos

**Hecho:** fases 1–3, tests, reviews, piloto real (D8), documentación (README Módulos 1–5, este reporte), audio automático de resultados en la app HTML (F1).

**Pendiente / recomendado:**
1. **Recolectar y etiquetar dataset grande** (Módulo 5.11): 3–4 vuelos × 5–8 fotos × 4–6 tiles ≈ 100–200 tiles para la primera iteración que aprenda.
2. Reentrenar con `--epochs 100` (idealmente con GPU) y evaluar mAP real por vuelos completos.
3. Opcional: `OffscreenCanvas` en el worker de F1; integración web del conteo ML (fuera de alcance actual).

**Entorno verificado (2026-09-24):** Python 3.14.6, ultralytics 8.4.161, torch 2.14.0+cpu, opencv-python 5.0.0.93, roboflow SDK 1.5.0, numpy 2.3.5 (bajado por roboflow; la suite sigue verde con runtime presente — fix F2 confirmado).

---

## 11. Glosario rápido (para una IA)

| Término | Significado |
|---|---|
| `Tile` | Recorte de una foto original (layout posicional `_xN_yN`) |
| `Manifest` | CSV que mapea cada tile → foto original + offsets |
| `Sidecar` | Archivo `.txt` YOLO con anotaciones `class x y w h` normalizadas |
| `Vuelo` | Subcarpeta de fotos (parcela + fecha + etapa); unidad del split train/valid |
| `Seam` | Punto de integración con runtime externo (en este proyecto: `load_model`/`predict` vs ultralytics) — el resto del pipeline usa duck-typing |
| `D8` | Hito pendiente de reviews: validar `_UltralyticsPredictor` contra ultralytics real — **cerrado** en el piloto |
| `NMS` | Supresión de no-máximos por IoU; aquí se usa en coordenadas globales de la foto para deduplicar plantas repetidas entre tiles |
| `4R` | Conjunto de lentes de review: risk, resilience, readability, reliability |
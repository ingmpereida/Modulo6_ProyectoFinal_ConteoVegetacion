# Conteo de Vegetación con IA — Documentación del Proyecto

Proyecto final del Módulo 6: pipeline de visión por computadora para **contar plantas en parcelas agrícolas** a partir de fotos de dron o celular. El objetivo del sistema es estimar la densidad de vegetación (especialmente en etapas de emergencia, donde las plantas son pequeñas y dispersas), primero con procesamiento clásico de color y, en fases posteriores, con un modelo entrenado.

Este README está escrito para que **una IA o una persona pueda entender el proyecto y continuar el trabajo sin re-descubrir nada**. Léelo completo antes de modificar código.

---

## Estado actual (resumen ejecutivo)

| Archivo | Rol | Estado |
|---|---|---|
| `conteo_vegetacion.html` | Fase 1 — prototipo de conteo clásico por color, 100 % en el navegador | Funcional, con worker |
| `tile_pipeline.py` | Fase 2 — preparación de imágenes para etiquetado (tiles + manifest) | Funcional |
| `requirements.txt` | Dependencias Python (`Pillow`, `numpy`) | Creado |
| `tests/computeDetection.test.js` | Test automatizado de `computeDetection` (runner nativo de Node) | 8/8 pasando |
| `package.json` | Script `npm test` para correr los tests | — |
| `README.md` | Este documento | — |

El directorio **no es un repositorio git** todavía (por lo tanto no tiene `.gitignore`); la lógica de detección sí tiene tests automatizados — ver *Tests automatizados*.

---

## Pipeline global (fases del proyecto)

```
Fase 1 (hecha)        Fase 2 (hecha)              Fase 2–3 (pendiente)
Fotos de dron     ->  Cortar en tiles          ->  Etiquetar en Roboflow/CVAT
conteo clásico        descartar tiles vacíos       entrenar modelo (YOLO/CNN)
por color             generar manifest.csv         conteo con mayor precisión
```

- **Fase 1**: `conteo_vegetacion.html` — conteo por color (Excess Green Index) + componentes conexas. Prototipo interactivo.
- **Fase 2**: `tile_pipeline.py` — convierte fotos grandes en tiles etiquetables, descartando suelo vacío.
- **Fase 2–3 (futura)**: las propuestas mencionan un modelo entrenado para copas solapadas y alta densidad. Los tiles de la Fase 2 alimentan exactamente ese etiquetado.

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

## Flujo de datos completo (cómo encajan las piezas)

```
Foto de dron (celular/dron/ortomosaico)
  │
  ├─► Fase 1: conteo_vegetacion.html (humano, en el navegador)
  │     plan → preview → ajustar sensibilidad → imagen original marcada + conteo
  │
  └─► Fase 2: tile_pipeline.py (preparación para ML)
        fotos → tiles etiquetables + manifest.csv
        → (futuro) Roboflow/CVAT → modelo entrenado → Fase 1 con IA
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
| Commits | Conventional Commits (no existe repo todavía; respetar al inicializarlo) |

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

1. Inicializar **git** (repo + commits conventionales) y añadir `.gitignore`.
2. **Fase 2–3**: etiquetar tiles en Roboflow/CVAT usando el `manifest.csv` y entrenar el modelo para copas solapadas.
3. Optimización opcional: `OffscreenCanvas` dentro del worker para evitar el `getImageData`/transferencia en el hilo principal.

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

## Relación con la memoria persistente (Engram)

Si esta sesión o una futura tiene acceso al servidor MCP de **Engram** (proyecto `proyectofinal-modulo6`), las decisiones no obvias también están guardadas como observaciones (`discovery/`, `bug/…`, `architecture/web-worker…`). En caso de duda entre este README y una observación, **el código y este README mandan**; Engram es contexto complementario.
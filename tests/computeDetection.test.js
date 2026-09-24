/**
 * Test automatizado de computeDetection (conteo_vegetacion.html).
 *
 * La funcion de deteccion es pura: no toca DOM ni canvas, vive embebida en el
 * HTML. Este test la extrae del archivo y la evalua en Node, de la misma forma
 * que la app serializa la funcion con toString() para construir su web worker.
 *
 * Correr con:
 *   node --test tests/
 *   (o: npm test)
 */
'use strict';

const { test } = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');

const PROJ_ROOT = path.join(__dirname, '..');
const HTML_PATH = path.join(PROJ_ROOT, 'conteo_vegetacion.html');
const HTML = fs.readFileSync(HTML_PATH, 'utf8');

/** Extrae `function <name>(...) { ... }` del codigo fuente contando llaves. */
function extractFunction(source, name) {
  const start = source.indexOf(`function ${name}(`);
  if (start === -1) {
    throw new Error(`function ${name} not found in conteo_vegetacion.html`);
  }
  const bodyStart = source.indexOf('{', start) + 1;
  let depth = 1;
  let i = bodyStart;
  while (i < source.length && depth > 0) {
    if (source[i] === '{') depth++;
    else if (source[i] === '}') depth--;
    i++;
  }
  if (depth !== 0) {
    throw new Error(`unbalanced braces while extracting ${name}`);
  }
  return source.slice(start, i);
}

const computeDetection = eval(`(${extractFunction(HTML, 'computeDetection')})`);

/**
 * Genera una imagen sintetica RGB.
 * @param {number} w ancho en pixeles
 * @param {number} h alto en pixeles
 * @param {Array<[number,number,number,number]>} blobs [x0,y0,x1,y1) verdes
 */
function makeImage(w, h, blobs) {
  const data = new Uint8ClampedArray(w * h * 4);
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      const i = (y * w + x) * 4;
      const inBlob = blobs.some(([x0, y0, x1, y1]) => x >= x0 && x < x1 && y >= y0 && y < y1);
      data[i]     = inBlob ? 40 : 120;   // R
      data[i + 1] = inBlob ? 160 : 120;  // G
      data[i + 2] = inBlob ? 40 : 120;   // B
      data[i + 3] = 255;                 // A
    }
  }
  return data;
}

const THRESHOLD = 18; // default de la app
const SPLIT_OFF = false;

test('extrae computeDetection del HTML', () => {
  assert.equal(typeof computeDetection, 'function');
});

test('detecta un blob verde aislado', () => {
  const w = 30, h = 20;
  const data = makeImage(w, h, [[12, 7, 18, 13]]); // 6x6 = 36 px
  const res = computeDetection(w, h, data, THRESHOLD, 5, SPLIT_OFF);
  assert.equal(res.components, 1);
  assert.equal(res.markers.length, 1);
  assert.equal(res.greenPixels, 36);
});

test('respeta el tamano minimo (minSize)', () => {
  const w = 30, h = 20;
  const data = makeImage(w, h, [[12, 7, 18, 13]]);
  const res = computeDetection(w, h, data, THRESHOLD, 1000, SPLIT_OFF); // blob mas chico que minSize
  assert.equal(res.components, 0);
  assert.equal(res.markers.length, 0);
});

test('detecta dos blobs separados como dos plantas', () => {
  const w = 40, h = 40;
  const data = makeImage(w, h, [[5, 5, 9, 9], [25, 25, 29, 29]]); // 4x4 c/u = 16 px
  const res = computeDetection(w, h, data, THRESHOLD, 5, SPLIT_OFF);
  assert.equal(res.components, 2);
  assert.equal(res.markers.length, 2);
  assert.equal(res.greenPixels, 32);
});

test('detecta un blob pegado al borde de la imagen', () => {
  const w = 30, h = 30;
  const data = makeImage(w, h, [[0, 0, 6, 6]]); // esquina sup-izq
  const res = computeDetection(w, h, data, THRESHOLD, 5, SPLIT_OFF);
  assert.equal(res.components, 1);
  assert.equal(res.markers.length, 1);
});

test('una imagen sin verde no produce ningun marcador', () => {
  const w = 30, h = 30;
  const data = makeImage(w, h, []);
  const res = computeDetection(w, h, data, THRESHOLD, 5, SPLIT_OFF);
  assert.equal(res.components, 0);
  assert.equal(res.markers.length, 0);
  assert.equal(res.greenPixels, 0);
});

test('splitClusters divide una mancha grande segun la mediana', () => {
  // 3 blobs chicos de 36 px + 1 grande de 900 px -> mediana de areas = 36
  const w = 80, h = 80;
  const data = makeImage(w, h, [
    [5, 5, 11, 11],     // 36 px
    [5, 20, 11, 26],    // 36 px
    [5, 35, 11, 41],    // 36 px
    [40, 10, 70, 40],   // 900 px
  ]);
  const withSplit = computeDetection(w, h, data, THRESHOLD, 5, true);
  assert.equal(withSplit.components, 4);
  assert.equal(withSplit.markers.length, 28); // 3 + round(900/36)=25
  assert.equal(withSplit.greenPixels, 1008);  // 3*36 + 900

  const withoutSplit = computeDetection(w, h, data, THRESHOLD, 5, SPLIT_OFF);
  assert.equal(withoutSplit.markers.length, 4);
  assert.equal(withoutSplit.markers.every((m) => m.split === false), true);
});

test('la mediana de areas no depende del orden de deteccion (determinismo)', () => {
  const w = 80, h = 80;
  const data = makeImage(w, h, [
    [5, 5, 11, 11],
    [5, 20, 11, 26],
    [5, 35, 11, 41],
    [40, 10, 70, 40],
  ]);
  const a = computeDetection(w, h, data, THRESHOLD, 5, true);
  const b = computeDetection(w, h, data, THRESHOLD, 5, true);
  assert.deepEqual(a.markers, b.markers);
  assert.equal(a.greenPixels, b.greenPixels);
});
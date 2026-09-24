/**
 * Test automatizado del audio hablado (conteo_vegetacion.html).
 *
 * buildResultSpeech es la funcion pura que genera la frase leida en voz alta al
 * finalizar el conteo. Se extrae del HTML con la misma tecnica de conteo de
 * llaves que usa tests/computeDetection.test.js (y que la app usa para
 * serializar computeDetection en su web worker) y se evalua en Node sin DOM.
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

const buildResultSpeech = eval(`(${extractFunction(HTML, 'buildResultSpeech')})`);

test('extrae buildResultSpeech del HTML', () => {
  assert.equal(typeof buildResultSpeech, 'function');
});

test('frase completa con coma decimal en el porcentaje', () => {
  const speech = buildResultSpeech(25, 12, '34.5%');
  assert.equal(
    speech,
    'Conteo de vegetación finalizado. Plantas detectadas: 25. Manchas analizadas: 12. Porcentaje de cobertura verde: 34,5 por ciento.'
  );
});

test('cero detecciones se lee correctamente', () => {
  const speech = buildResultSpeech(0, 0, '0.0%');
  assert.ok(speech.includes('Plantas detectadas: 0.'));
  assert.ok(speech.includes('Manchas analizadas: 0.'));
  assert.ok(speech.includes('0,0 por ciento'));
});

test('porcentaje no numerico cae en "sin dato" en vez de texto crudo', () => {
  const speech = buildResultSpeech(3, 2, '—');
  assert.ok(speech.includes('Porcentaje de cobertura verde: sin dato.'));
  assert.ok(!speech.includes('—'));
});
/**
 * Extract training features using Meyda — identical logic to App.js.
 * This guarantees JS inference matches training exactly.
 *
 * Usage: node extract.js
 * Output: ../features_meyda.json  (features + labels array)
 */

const fs   = require('fs');
const path = require('path');
const wav  = require('node-wav');
const Meyda = require('meyda');

// ── Parameters (must match App.js) ───────────────────────────────────────────
const SR           = 32000;
const WIN_SAMPLES  = Math.round(SR * 0.3);   // 9600
const STRIDE_SAMPLES = Math.round(SR * 0.2); // 6400
const N_MFCC       = 40;
const FFT_SIZE     = 2048;
const HOP_SIZE     = 512;

const BASE_DIR      = path.resolve(__dirname, '..', '..', '..', 'project-pobblebonk', 'backend');
const TRAINING_DIR  = path.join(BASE_DIR, 'Training Audio');
const BACKGROUND_DIR = path.join(BASE_DIR, 'Background Audio');
const OUTPUT_FILE   = path.join(__dirname, '..', 'features_meyda.json');

// Initialise Meyda globals once before any extraction
Meyda.sampleRate = SR;
Meyda.numberOfMFCCCoefficients = N_MFCC;

// ── Feature helpers (identical to App.js) ────────────────────────────────────

function colMean(matrix) {
  if (!matrix.length) return [];
  const k = matrix[0].length;
  const mean = new Array(k).fill(0);
  for (const row of matrix) for (let j = 0; j < k; j++) mean[j] += row[j];
  return mean.map(v => v / matrix.length);
}

function colStd(matrix, mean) {
  if (!matrix.length) return new Array(mean.length).fill(0);
  const variance = new Array(mean.length).fill(0);
  for (const row of matrix) for (let j = 0; j < mean.length; j++) {
    const d = row[j] - mean[j]; variance[j] += d * d;
  }
  return variance.map(v => Math.sqrt(v / matrix.length));
}

// librosa-compatible delta (width=9, D=4)
function computeDelta(matrix) {
  const D = 4;
  const norm = 2 * (D * (D + 1) * (2 * D + 1)) / 6; // 60
  const n = matrix.length;
  const k = matrix[0].length;
  return matrix.map((_, i) => {
    const d = new Array(k).fill(0);
    for (let w = 1; w <= D; w++) {
      const prev = matrix[Math.max(0, i - w)];
      const next = matrix[Math.min(n - 1, i + w)];
      for (let j = 0; j < k; j++) d[j] += w * (next[j] - prev[j]);
    }
    return d.map(v => v / norm);
  });
}

function computeContrastFrames(segment) {
  const nBands = 6;
  const quantile = 0.02;
  const edgesHz = [0];
  for (let b = 0; b <= nBands; b++) edgesHz.push(200 * Math.pow(2, b));
  edgesHz.push(SR / 2);

  const frames = [];
  for (let i = 0; i + FFT_SIZE <= segment.length; i += HOP_SIZE) {
    const frame = segment.slice(i, i + FFT_SIZE);
    const power = Meyda.extract('powerSpectrum', frame);
    if (!power) { frames.push(new Array(nBands + 1).fill(0)); continue; }
    const mags = power.map(Math.sqrt);
    const nBins = mags.length;
    const edgesBins = edgesHz.map(f => Math.min(nBins, Math.round(f * FFT_SIZE / SR)));

    const contrast = [];
    for (let b = 0; b <= nBands; b++) {
      const band = Array.from(mags.slice(edgesBins[b], edgesBins[b + 1])).sort((a, b) => a - b);
      if (!band.length) { contrast.push(0); continue; }
      const nQ = Math.max(1, Math.round(quantile * band.length));
      const valley = band.slice(0, nQ).reduce((s, v) => s + v, 0) / nQ;
      const peak   = band.slice(-nQ).reduce((s, v) => s + v, 0) / nQ;
      contrast.push(10 * Math.log10((peak + 1e-10) / (valley + 1e-10)));
    }
    frames.push(contrast);
  }
  return frames;
}

function extractFeatures(segment) {
  const mfccFrames     = [];
  const centroidFrames = [];

  for (let i = 0; i + FFT_SIZE <= segment.length; i += HOP_SIZE) {
    const frame    = segment.slice(i, i + FFT_SIZE);
    const mfcc     = Meyda.extract('mfcc', frame);
    const centroid = Meyda.extract('spectralCentroid', frame);
    if (mfcc) mfccFrames.push(Array.from(mfcc));
    centroidFrames.push([centroid ?? 0]);
  }

  if (!mfccFrames.length) return null;

  const delta1 = computeDelta(mfccFrames);
  const delta2 = computeDelta(delta1);
  const contrastFrames = computeContrastFrames(segment);

  const mfccMean = colMean(mfccFrames);   const mfccStd = colStd(mfccFrames, mfccMean);
  const d1Mean   = colMean(delta1);       const d1Std   = colStd(delta1, d1Mean);
  const d2Mean   = colMean(delta2);       const d2Std   = colStd(delta2, d2Mean);
  const contMean = colMean(contrastFrames); const contStd = colStd(contrastFrames, contMean);
  const centMean = colMean(centroidFrames); const centStd = colStd(centroidFrames, centMean);

  return [
    ...mfccMean, ...mfccStd,
    ...d1Mean,   ...d1Std,
    ...d2Mean,   ...d2Std,
    ...contMean, ...contStd,
    ...centMean, ...centStd,
  ];
}

// ── Audio helpers ─────────────────────────────────────────────────────────────

/** Linear-interpolation resample — only needed when file SR ≠ 32000 */
function resample(data, sourceSr, targetSr) {
  if (sourceSr === targetSr) return data;
  const ratio     = sourceSr / targetSr;
  const newLength = Math.round(data.length / ratio);
  const result    = new Float32Array(newLength);
  for (let i = 0; i < newLength; i++) {
    const pos  = i * ratio;
    const idx  = Math.floor(pos);
    const frac = pos - idx;
    const a    = data[Math.min(idx,     data.length - 1)];
    const b    = data[Math.min(idx + 1, data.length - 1)];
    result[i]  = a + frac * (b - a);
  }
  return result;
}

function processFile(filePath, label, featuresArr, labelsArr) {
  let buffer;
  try { buffer = fs.readFileSync(filePath); }
  catch (e) { console.error(`  [SKIP] read error: ${filePath}: ${e.message}`); return 0; }

  let decoded;
  try { decoded = wav.decode(buffer); }
  catch (e) { console.error(`  [SKIP] wav decode error: ${filePath}: ${e.message}`); return 0; }

  // Use first channel only; resample if needed
  let audio = decoded.channelData[0];
  if (decoded.sampleRate !== SR) audio = resample(audio, decoded.sampleRate, SR);

  let count = 0;
  for (let start = 0; start + WIN_SAMPLES <= audio.length; start += STRIDE_SAMPLES) {
    const seg  = audio.slice(start, start + WIN_SAMPLES);
    const feat = extractFeatures(seg);
    if (feat) { featuresArr.push(feat); labelsArr.push(label); count++; }
  }
  return count;
}

// ── Main ──────────────────────────────────────────────────────────────────────

const features = [];
const labels   = [];

console.log('Extracting training audio windows...');
const species = fs.readdirSync(TRAINING_DIR).sort();
for (const sp of species) {
  const spDir = path.join(TRAINING_DIR, sp);
  if (!fs.statSync(spDir).isDirectory()) continue;
  process.stdout.write(`  ${sp} ... `);
  const files = fs.readdirSync(spDir).filter(f => f.toLowerCase().endsWith('.wav'));
  let total = 0;
  for (const f of files) total += processFile(path.join(spDir, f), sp, features, labels);
  console.log(`${total} windows`);
}

console.log('\nExtracting background audio windows...');
if (fs.existsSync(BACKGROUND_DIR)) {
  const files = fs.readdirSync(BACKGROUND_DIR).filter(f => f.toLowerCase().endsWith('.wav'));
  let total = 0;
  for (const f of files) total += processFile(path.join(BACKGROUND_DIR, f), 'Background', features, labels);
  console.log(`  ${total} background windows`);
}

const nFeatures = features[0]?.length ?? 0;
console.log(`\nTotal: ${features.length} samples, ${nFeatures} features`);

fs.writeFileSync(OUTPUT_FILE, JSON.stringify({ features, labels }));
console.log(`Saved → ${OUTPUT_FILE}`);

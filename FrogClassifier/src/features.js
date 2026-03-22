import Meyda from 'meyda';
import { SR, N_MFCC, FFT_SIZE, HOP_SIZE } from './constants';

const N_BANDS  = 6;
const QUANTILE = 0.02;

// Contrast band edges — same octave structure as training, computed once
const edgesHz = [0];
for (let b = 0; b <= N_BANDS; b++) edgesHz.push(200 * Math.pow(2, b));
edgesHz.push(SR / 2);
const N_BINS    = FFT_SIZE / 2 + 1; // Meyda powerSpectrum length = 1025
const EDGES_BINS = edgesHz.map(f => Math.min(N_BINS, Math.round(f * FFT_SIZE / SR)));

// ---------------------------------------------------------------------------
// Matrix helpers — Float64Array accumulators reduce GC pressure
// ---------------------------------------------------------------------------
function colMean(matrix) {
  if (!matrix.length) return [];
  const k   = matrix[0].length;
  const acc = new Float64Array(k);
  for (const row of matrix) for (let j = 0; j < k; j++) acc[j] += row[j];
  return Array.from(acc, v => v / matrix.length);
}

function colStd(matrix, mean) {
  if (!matrix.length) return new Array(mean.length).fill(0);
  const k   = mean.length;
  const acc = new Float64Array(k);
  for (const row of matrix) for (let j = 0; j < k; j++) { const d = row[j] - mean[j]; acc[j] += d * d; }
  return Array.from(acc, v => Math.sqrt(v / matrix.length));
}

function computeDelta(matrix) {
  if (!matrix.length) return [];
  const D = 4, norm = 60;
  const n = matrix.length, k = matrix[0].length;
  return matrix.map((_, i) => {
    const d = new Float64Array(k);
    for (let w = 1; w <= D; w++) {
      const prev = matrix[Math.max(0, i - w)];
      const next = matrix[Math.min(n - 1, i + w)];
      for (let j = 0; j < k; j++) d[j] += w * (next[j] - prev[j]);
    }
    return Array.from(d, v => v / norm);
  });
}

// mags must be a Float32Array (power.map(Math.sqrt)) to preserve the float32
// precision used during training — do not change to Math.sqrt(power[k]).
function contrastFromMags(mags) {
  const contrast = [];
  for (let b = 0; b <= N_BANDS; b++) {
    const band = Array.from(mags.slice(EDGES_BINS[b], EDGES_BINS[b + 1])).sort((a, b) => a - b);
    if (!band.length) { contrast.push(0); continue; }
    const nQ    = Math.max(1, Math.round(QUANTILE * band.length));
    const valley = band.slice(0, nQ).reduce((s, v) => s + v, 0) / nQ;
    const peak   = band.slice(-nQ).reduce((s, v) => s + v, 0) / nQ;
    contrast.push(10 * Math.log10((peak + 1e-10) / (valley + 1e-10)));
  }
  return contrast;
}

// ---------------------------------------------------------------------------
// extractFeatures — single Meyda call per hop shares one FFT across mfcc,
// spectralCentroid, and powerSpectrum (~3x fewer FFTs vs separate calls).
// ---------------------------------------------------------------------------
export function extractFeatures(segment) {
  Meyda.sampleRate = SR;
  Meyda.numberOfMFCCCoefficients = N_MFCC;

  const mfccFrames     = [];
  const centroidFrames = [];
  const contrastFrames = [];

  for (let i = 0; i + FFT_SIZE <= segment.length; i += HOP_SIZE) {
    const frame  = segment.slice(i, i + FFT_SIZE);
    const result = Meyda.extract(['mfcc', 'spectralCentroid', 'powerSpectrum'], frame);

    if (result.mfcc) mfccFrames.push(Array.from(result.mfcc));
    centroidFrames.push([result.spectralCentroid ?? 0]);
    contrastFrames.push(
      result.powerSpectrum
        ? contrastFromMags(result.powerSpectrum.map(Math.sqrt))
        : new Array(N_BANDS + 1).fill(0)
    );
  }

  const delta1 = computeDelta(mfccFrames);
  const delta2 = computeDelta(delta1);

  const mfccMean = colMean(mfccFrames);    const mfccStd = colStd(mfccFrames, mfccMean);
  const d1Mean   = colMean(delta1);         const d1Std   = colStd(delta1, d1Mean);
  const d2Mean   = colMean(delta2);         const d2Std   = colStd(delta2, d2Mean);
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

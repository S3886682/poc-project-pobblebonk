/**
 * Test the SVM model locally using the exact same feature + inference
 * code as App.js.  Supports WAV and MP3 files.
 *
 * Usage:
 *   node test_model.js                         # all training + background audio
 *   node test_model.js --sample 5              # random 5 files per species
 *   node test_model.js --testdir <path>        # flat folder, label inferred from filename
 *
 * Confidence = window agreement: fraction of non-background windows
 * that voted for the winning species (same as App.js).
 */

'use strict';
const fs      = require('fs');
const os      = require('os');
const path    = require('path');
const wav     = require('node-wav');
const Meyda   = require('meyda');
const ffmpeg  = require('fluent-ffmpeg');
const ffmpegPath = require('@ffmpeg-installer/ffmpeg').path;
ffmpeg.setFfmpegPath(ffmpegPath);

// ── Config ────────────────────────────────────────────────────────────────────
const SR             = 32000;
const WIN_SAMPLES    = Math.round(SR * 0.3);
const STRIDE_SAMPLES = Math.round(SR * 0.2);
const N_MFCC         = 40;
const FFT_SIZE       = 2048;
const HOP_SIZE       = 512;

const BASE_DIR       = path.resolve(__dirname, '..', '..', '..', 'project-pobblebonk', 'backend');
const TRAINING_DIR   = path.join(BASE_DIR, 'Training Audio');
const BACKGROUND_DIR = path.join(BASE_DIR, 'Background Audio');
const MODEL_PATH     = path.join(__dirname, '..', 'svm_model.json');

Meyda.sampleRate = SR;
Meyda.numberOfMFCCCoefficients = N_MFCC;

const model = JSON.parse(fs.readFileSync(MODEL_PATH, 'utf8'));

// ── Audio loading (WAV + MP3 via ffmpeg) ──────────────────────────────────────

function loadWav(filePath) {
  const decoded = wav.decode(fs.readFileSync(filePath));
  let audio = decoded.channelData[0];
  if (decoded.sampleRate !== SR) audio = resample(audio, decoded.sampleRate, SR);
  return audio;
}

function convertToWav(filePath) {
  return new Promise((resolve, reject) => {
    const tmp = path.join(os.tmpdir(), `frog_test_${Date.now()}.wav`);
    ffmpeg(filePath)
      .audioFrequency(SR)
      .audioChannels(1)
      .audioBitrate(16)
      .toFormat('wav')
      .on('error', reject)
      .on('end', () => resolve(tmp))
      .save(tmp);
  });
}

async function loadAudio(filePath) {
  const ext = path.extname(filePath).toLowerCase();
  if (ext === '.wav') return loadWav(filePath);
  // Convert MP3/M4A/etc. to WAV via ffmpeg
  const tmpWav = await convertToWav(filePath);
  try {
    return loadWav(tmpWav);
  } finally {
    fs.unlinkSync(tmpWav);
  }
}

function resample(data, sourceSr, targetSr) {
  if (sourceSr === targetSr) return data;
  const ratio = sourceSr / targetSr;
  const newLength = Math.round(data.length / ratio);
  const result = new Float32Array(newLength);
  for (let i = 0; i < newLength; i++) {
    const pos = i * ratio, idx = Math.floor(pos), frac = pos - idx;
    result[i] = data[Math.min(idx, data.length-1)] * (1-frac)
              + data[Math.min(idx+1, data.length-1)] * frac;
  }
  return result;
}

// ── Feature extraction (identical to App.js) ──────────────────────────────────
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
function computeDelta(matrix) {
  const D = 4, norm = 60, n = matrix.length, k = matrix[0].length;
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
  const nBands = 6, quantile = 0.02;
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
  const mfccFrames = [], centroidFrames = [];
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
  const mfccMean = colMean(mfccFrames);     const mfccStd = colStd(mfccFrames, mfccMean);
  const d1Mean   = colMean(delta1);         const d1Std   = colStd(delta1, d1Mean);
  const d2Mean   = colMean(delta2);         const d2Std   = colStd(delta2, d2Mean);
  const contMean = colMean(contrastFrames); const contStd = colStd(contrastFrames, contMean);
  const centMean = colMean(centroidFrames); const centStd = colStd(centroidFrames, centMean);
  return [...mfccMean, ...mfccStd, ...d1Mean, ...d1Std, ...d2Mean, ...d2Std,
          ...contMean, ...contStd, ...centMean, ...centStd];
}

// ── SVM inference (identical to App.js) ───────────────────────────────────────
function rbfKernel(x, y, gamma) {
  let norm = 0;
  for (let i = 0; i < x.length; i++) { const d = x[i] - y[i]; norm += d * d; }
  return Math.exp(-gamma * norm);
}
function predict(features) {
  const scaled = features.map((v, i) => (v - model.scaler_mean[i]) / model.scaler_scale[i]);
  const nClasses = model.classes.length;
  const votes = new Array(nClasses).fill(0);
  const kernelVals = model.support_vectors.map(sv => rbfKernel(scaled, sv, model.gamma));
  const svStart = new Array(nClasses).fill(0);
  for (let c = 1; c < nClasses; c++) svStart[c] = svStart[c - 1] + model.n_support[c - 1];
  let pairIdx = 0;
  for (let i = 0; i < nClasses; i++) {
    for (let j = i + 1; j < nClasses; j++) {
      let sum = model.intercept[pairIdx];
      for (let s = 0; s < model.n_support[i]; s++)
        sum += model.dual_coef[j - 1][svStart[i] + s] * kernelVals[svStart[i] + s];
      for (let s = 0; s < model.n_support[j]; s++)
        sum += model.dual_coef[i][svStart[j] + s] * kernelVals[svStart[j] + s];
      if (sum > 0) votes[i]++; else votes[j]++;
      pairIdx++;
    }
  }
  const winnerIdx = votes.indexOf(Math.max(...votes));
  return model.classes[winnerIdx]; // just the label; confidence computed from window agreement
}

// ── Classify a single file ─────────────────────────────────────────────────────
async function classifyFile(filePath) {
  let audio;
  try { audio = await loadAudio(filePath); }
  catch (e) { return { label: '(load error)', confidence: 0, windows: 0, bgWindows: 0 }; }

  const allPreds = [];
  for (let start = 0; start + WIN_SAMPLES <= audio.length; start += STRIDE_SAMPLES) {
    const seg  = audio.slice(start, start + WIN_SAMPLES);
    const feat = extractFeatures(seg);
    if (!feat || feat.some(v => !isFinite(v))) continue;
    allPreds.push(predict(feat));
  }

  if (!allPreds.length) return { label: '(too short)', confidence: 0, windows: 0, bgWindows: 0 };

  const nonBg = allPreds.filter(l => l !== 'Background');
  const bgWindows = allPreds.length - nonBg.length;

  if (!nonBg.length) {
    return { label: 'Background', confidence: 1, windows: allPreds.length, bgWindows };
  }

  // Window agreement: count votes per species among non-background windows
  const counts = {};
  for (const label of nonBg) counts[label] = (counts[label] || 0) + 1;
  const winner = Object.entries(counts).sort((a, b) => b[1] - a[1])[0][0];
  const confidence = counts[winner] / nonBg.length; // window agreement fraction

  return { label: winner, confidence, windows: allPreds.length, bgWindows };
}

// ── Test runner ───────────────────────────────────────────────────────────────
const args     = process.argv.slice(2);
const sample   = args.includes('--sample')  ? parseInt(args[args.indexOf('--sample') + 1])  : Infinity;
const testDir  = args.includes('--testdir') ? args[args.indexOf('--testdir') + 1] : null;

function shuffle(arr) {
  for (let i = arr.length - 1; i > 0; i--) {
    const j = Math.floor(Math.random() * (i + 1));
    [arr[i], arr[j]] = [arr[j], arr[i]];
  }
  return arr;
}

const AUDIO_EXTS = new Set(['.wav', '.mp3', '.m4a', '.ogg', '.flac']);

async function testFiles(files, expectedLabel, dir) {
  let correct = 0;
  for (const f of files) {
    const { label, confidence, windows, bgWindows } = await classifyFile(path.join(dir, f));
    const ok = label === expectedLabel;
    const confPct = Math.round(confidence * 100);
    const tick = ok ? '✓' : '✗';
    console.log(`  ${tick} ${f.padEnd(45)} → ${label.padEnd(35)} (${confPct}% agree, ${windows}w, ${bgWindows} bg)`);
    if (ok) correct++;
  }
  return { correct, total: files.length };
}

async function main() {
  let totalCorrect = 0, totalFiles = 0;
  const perClass = {};

  if (testDir) {
    // ── Flat folder mode: label inferred from filename (e.g. "09 Green Tree Frog.mp3") ──
    console.log(`\nTesting folder: ${testDir}\n`);
    const files = fs.readdirSync(testDir)
      .filter(f => AUDIO_EXTS.has(path.extname(f).toLowerCase()))
      .sort();

    for (const f of files) {
      // Strip leading "NN " number prefix and extension to get label
      const base = path.basename(f, path.extname(f)).replace(/^\d+\s+/, '');
      const { label, confidence, windows, bgWindows } = await classifyFile(path.join(testDir, f));
      const confPct = Math.round(confidence * 100);
      console.log(`  ${f.padEnd(45)} → ${label.padEnd(35)} (${confPct}% agree, ${windows}w, ${bgWindows} bg)`);
      console.log(`    Expected: ${base}`);
      console.log('');
    }
  } else {
    // ── Training + background folder mode ──────────────────────────────────────
    const species = fs.readdirSync(TRAINING_DIR).sort();
    for (const sp of species) {
      const spDir = path.join(TRAINING_DIR, sp);
      if (!fs.statSync(spDir).isDirectory()) continue;
      let files = fs.readdirSync(spDir).filter(f => AUDIO_EXTS.has(path.extname(f).toLowerCase()));
      if (files.length > sample) files = shuffle(files).slice(0, sample);
      console.log(`\n── ${sp} ──`);
      const { correct, total } = await testFiles(files, sp, spDir);
      perClass[sp] = { correct, total };
      totalCorrect += correct; totalFiles += total;
    }

    console.log('\n── Background ──');
    let bgFiles = fs.readdirSync(BACKGROUND_DIR).filter(f => AUDIO_EXTS.has(path.extname(f).toLowerCase()));
    if (bgFiles.length > sample) bgFiles = shuffle(bgFiles).slice(0, sample);
    const { correct: bgC, total: bgT } = await testFiles(bgFiles, 'Background', BACKGROUND_DIR);
    perClass['Background'] = { correct: bgC, total: bgT };
    totalCorrect += bgC; totalFiles += bgT;

    console.log('\n═══════════════════════════════════════════════════════════');
    console.log(`Overall accuracy: ${totalCorrect}/${totalFiles} = ${(totalCorrect/totalFiles*100).toFixed(1)}%`);
    console.log('\nPer-class accuracy:');
    for (const [label, { correct: c, total: t }] of Object.entries(perClass).sort()) {
      const pct = (c / t * 100).toFixed(0).padStart(3);
      const bar = '█'.repeat(Math.round(c/t*20)).padEnd(20, '░');
      console.log(`  ${label.padEnd(40)} ${bar} ${pct}% (${c}/${t})`);
    }
  }
}

main().catch(console.error);

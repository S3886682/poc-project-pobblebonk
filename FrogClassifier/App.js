import { useState } from 'react';
import { View, Text, Button, StyleSheet } from 'react-native';
import { useAudioRecorder, AudioModule } from 'expo-audio';
import * as FileSystem from 'expo-file-system/legacy';
import * as DocumentPicker from 'expo-document-picker';
import { Buffer } from 'buffer';
import Meyda from 'meyda';

const model = require('./assets/svm_model.json');

// Match server.py parameters exactly
const SR = 32000;
const WIN_SAMPLES = Math.round(SR * 0.3);     // 9600
const STRIDE_SAMPLES = Math.round(SR * 0.2);  // 6400
const N_MFCC = 40;
const FFT_SIZE = 2048;
const HOP_SIZE = 512;
// WAV_HEADER_BYTES is no longer hardcoded — we parse the RIFF header properly

const RECORDING_OPTIONS = {
  extension: '.wav',
  sampleRate: SR,
  numberOfChannels: 1,
  bitRate: SR * 16,
  ios: {
    outputFormat: 'lpcm',
    audioQuality: 127,
    linearPCMBitDepth: 16,
    linearPCMIsBigEndian: false,
    linearPCMIsFloat: false,
  },
  android: {
    extension: '.wav',
    outputFormat: 'DEFAULT',
    audioEncoder: 'DEFAULT',
    sampleRate: SR,
    numberOfChannels: 1,
    bitRate: SR * 16,
  },
};

// --- SVM inference ---
function rbfKernel(x, y, gamma) {
  let norm = 0;
  for (let i = 0; i < x.length; i++) { const d = x[i] - y[i]; norm += d * d; }
  return Math.exp(-gamma * norm);
}

function predict(features) {
  const scaled = features.map((v, i) => (v - model.scaler_mean[i]) / model.scaler_scale[i]);
  const nClasses = model.classes.length;
  const votes = new Array(nClasses).fill(0);

  // Precompute kernel values for all support vectors once
  const kernelVals = model.support_vectors.map(sv => rbfKernel(scaled, sv, model.gamma));

  // Build SV class start indices from n_support
  const svStart = new Array(nClasses).fill(0);
  for (let c = 1; c < nClasses; c++) svStart[c] = svStart[c - 1] + model.n_support[c - 1];

  // OVO: one binary classifier per pair (i, j), i < j
  let pairIdx = 0;
  for (let i = 0; i < nClasses; i++) {
    for (let j = i + 1; j < nClasses; j++) {
      let sum = model.intercept[pairIdx];

      // SVs belonging to class i: dual_coef row is j-1
      for (let s = 0; s < model.n_support[i]; s++) {
        sum += model.dual_coef[j - 1][svStart[i] + s] * kernelVals[svStart[i] + s];
      }
      // SVs belonging to class j: dual_coef row is i
      for (let s = 0; s < model.n_support[j]; s++) {
        sum += model.dual_coef[i][svStart[j] + s] * kernelVals[svStart[j] + s];
      }

      if (sum > 0) votes[i]++; else votes[j]++;
      pairIdx++;
    }
  }

  const winnerIdx = votes.indexOf(Math.max(...votes));
  // Confidence = fraction of pairwise votes won by the winner (each class faces n-1 opponents)
  const confidence = votes[winnerIdx] / (nClasses - 1);
  return { label: model.classes[winnerIdx], confidence };
}

// --- Feature helpers ---

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
  for (const row of matrix) for (let j = 0; j < mean.length; j++) { const d = row[j] - mean[j]; variance[j] += d * d; }
  return variance.map(v => Math.sqrt(v / matrix.length));
}

// librosa-compatible delta (width=9, D=4)
function computeDelta(matrix) {
  const D = 4;
  const norm = 2 * (D * (D + 1) * (2 * D + 1)) / 6; // = 60
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

// librosa spectral_contrast defaults: n_bands=6, fmin=200, quantile=0.02
function computeContrastFrames(segment) {
  const nBands = 6;
  const quantile = 0.02;
  // Octave band edges in Hz: [0, 200, 400, 800, 1600, 3200, 6400, 12800, SR/2]
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
      const peak = band.slice(-nQ).reduce((s, v) => s + v, 0) / nQ;
      contrast.push(10 * Math.log10((peak + 1e-10) / (valley + 1e-10)));
    }
    frames.push(contrast);
  }
  return frames;
}

// Extract 616 features from a 0.3s segment, matching server.py
let _meydaLogged = false;
function extractFeatures(segment) {
  Meyda.sampleRate = SR;
  Meyda.numberOfMFCCCoefficients = N_MFCC;

  const mfccFrames = [];
  const centroidFrames = [];
  for (let i = 0; i + FFT_SIZE <= segment.length; i += HOP_SIZE) {
    const frame = segment.slice(i, i + FFT_SIZE);
    const mfcc = Meyda.extract('mfcc', frame);
    const centroid = Meyda.extract('spectralCentroid', frame);
    if (!_meydaLogged && mfcc) {
      console.log('[MEYDA] numberOfMelBands:', Meyda.numberOfMelBands);
      console.log('[MEYDA] numberOfMFCCCoefficients:', Meyda.numberOfMFCCCoefficients);
      console.log('[MEYDA] mfcc.length:', mfcc.length);
      console.log('[MEYDA] mfcc type:', Object.prototype.toString.call(mfcc));
      _meydaLogged = true;
    }
    if (mfcc) mfccFrames.push(Array.from(mfcc));
    centroidFrames.push([centroid ?? 0]);
  }

  const delta1 = computeDelta(mfccFrames);
  const delta2 = computeDelta(delta1);
  const contrastFrames = computeContrastFrames(segment);

  const mfccMean = colMean(mfccFrames);   const mfccStd = colStd(mfccFrames, mfccMean);
  const d1Mean = colMean(delta1);         const d1Std = colStd(delta1, d1Mean);
  const d2Mean = colMean(delta2);         const d2Std = colStd(delta2, d2Mean);
  const contMean = colMean(contrastFrames); const contStd = colStd(contrastFrames, contMean);
  const centMean = colMean(centroidFrames); const centStd = colStd(centroidFrames, centMean);

  // Matches server.py hstack order: mfcc, d1, d2, contrast, centroid × (mean, std)
  return [
    ...mfccMean, ...mfccStd,
    ...d1Mean,   ...d1Std,
    ...d2Mean,   ...d2Std,
    ...contMean, ...contStd,
    ...centMean, ...centStd,
  ];
}

function majorityVote(preds) {
  // preds is [{label, confidence}, ...]
  // confidence = window agreement: fraction of windows that voted for the winner
  const filtered = preds.filter(p => p.label !== 'Background');
  if (!filtered.length) {
    // All windows said Background — confidence = fraction of total that agreed
    return { label: 'Background', confidence: preds.length / preds.length };
  }
  const counts = {};
  for (const { label } of filtered) {
    counts[label] = (counts[label] || 0) + 1;
  }
  const winner = Object.entries(counts).sort((a, b) => b[1] - a[1])[0][0];
  // Window agreement: how many non-background windows chose the winner
  return { label: winner, confidence: counts[winner] / filtered.length };
}

// --- App ---
export default function App() {
  const audioRecorder = useAudioRecorder(RECORDING_OPTIONS);
  const [prediction, setPrediction] = useState(null);
  const [status, setStatus] = useState('');
  const [progress, setProgress] = useState(null); // null = hidden, 0-1 = classifying

  const startRecording = async () => {
    try {
      setStatus('Requesting permission...');
      console.log('[START] Requesting recording permissions...');
      const permResult = await AudioModule.requestRecordingPermissionsAsync();
      console.log('[START] Permission result:', JSON.stringify(permResult));
      const { granted } = permResult;
      if (!granted) { setStatus('Microphone permission denied'); return; }

      setStatus('Starting...');
      console.log('[START] Setting audio mode...');
      await AudioModule.setAudioModeAsync({ allowsRecording: true, playsInSilentMode: true });
      console.log('[START] Audio mode set. Calling prepareToRecordAsync...');
      await audioRecorder.prepareToRecordAsync();
      console.log('[START] prepareToRecordAsync done. audioRecorder state:', JSON.stringify({
        isRecording: audioRecorder.isRecording,
        uri: audioRecorder.uri,
      }));
      audioRecorder.record();
      console.log('[START] record() called. audioRecorder state:', JSON.stringify({
        isRecording: audioRecorder.isRecording,
        uri: audioRecorder.uri,
      }));
      setStatus('Recording');
    } catch (err) {
      setStatus(`Error: ${err.message}`);
      console.error('[START] Failed to start recording', err);
    }
  };

  const stopRecording = async () => {
    try {
      setStatus('Processing...');
      console.log('[STOP] stopRecording called. audioRecorder state:', JSON.stringify({
        isRecording: audioRecorder.isRecording,
        uri: audioRecorder.uri,
      }));
      console.log('[STOP] audioRecorder keys:', Object.keys(audioRecorder));

      let uri;
      try {
        console.log('[STOP] Calling audioRecorder.stop()...');
        const result = await audioRecorder.stop();
        console.log('[STOP] stop() returned:', JSON.stringify(result));
        console.log('[STOP] typeof result:', typeof result);
        uri = result?.uri ?? audioRecorder.uri;
        console.log('[STOP] uri from result:', uri);
      } catch (stopErr) {
        console.error('[STOP] stop() threw an error:', stopErr);
        console.error('[STOP] stop() error name:', stopErr?.name);
        console.error('[STOP] stop() error message:', stopErr?.message);
        console.error('[STOP] stop() error stack:', stopErr?.stack);
        uri = audioRecorder.uri;
        console.log('[STOP] Falling back to audioRecorder.uri:', uri);
      }

      if (!uri) {
        console.log('[STOP] uri still null, waiting 300ms...');
        await new Promise(res => setTimeout(res, 300));
        uri = audioRecorder.uri;
        console.log('[STOP] uri after wait:', uri);
      }

      if (!uri) { setStatus('Error: no recording URI after stop'); return; }
      await processAudio(uri);
    } catch (err) {
      setStatus(`Error: ${err.message}`);
      console.error('[STOP] Failed to stop recording', err);
      console.error('[STOP] error stack:', err?.stack);
    }
  };

  const processAudio = async (uri) => {
    setStatus('Reading file...');
    const base64 = await FileSystem.readAsStringAsync(uri, { encoding: 'base64' });
    const bytes = Buffer.from(base64, 'base64');

    setStatus('Decoding PCM...');
    // Parse RIFF/WAV header properly to find the data chunk offset,
    // sample rate, bit depth, and channel count — never assume 44 bytes.
    let dataOffset = -1, fileSr = SR, bitsPerSample = 16, numChannels = 1;
    try {
      let pos = 12; // skip RIFF(4) + fileSize(4) + WAVE(4)
      while (pos + 8 <= bytes.length) {
        const chunkId   = bytes.toString('ascii', pos, pos + 4);
        const chunkSize = bytes.readUInt32LE(pos + 4);
        if (chunkId === 'fmt ') {
          numChannels  = bytes.readUInt16LE(pos + 10);
          fileSr       = bytes.readUInt32LE(pos + 12);
          bitsPerSample = bytes.readUInt16LE(pos + 22);
        } else if (chunkId === 'data') {
          dataOffset = pos + 8;
          break;
        }
        pos += 8 + chunkSize + (chunkSize % 2); // chunks are word-aligned
      }
    } catch (_) {}

    if (dataOffset < 0) {
      setStatus('Error: could not parse WAV header');
      return;
    }
    if (bitsPerSample !== 16) {
      setStatus(`Error: unsupported bit depth (${bitsPerSample}-bit). Please use 16-bit WAV.`);
      return;
    }

    const bytesPerSample = bitsPerSample / 8;
    const nSamplesTotal  = Math.floor((bytes.length - dataOffset) / (bytesPerSample * numChannels));
    // Decode first channel only (mono mix)
    const monoData = new Float32Array(nSamplesTotal);
    for (let i = 0; i < nSamplesTotal; i++) {
      const offset = dataOffset + i * bytesPerSample * numChannels;
      monoData[i] = bytes.readInt16LE(offset) / 32768;
    }

    // Resample if the file isn't at the expected SR (linear interpolation)
    let audioData = monoData;
    if (fileSr !== SR) {
      console.log(`[WAV] Resampling from ${fileSr} Hz → ${SR} Hz`);
      const ratio = fileSr / SR;
      const newLen = Math.round(monoData.length / ratio);
      audioData = new Float32Array(newLen);
      for (let i = 0; i < newLen; i++) {
        const pos  = i * ratio;
        const idx  = Math.floor(pos);
        const frac = pos - idx;
        const a    = monoData[Math.min(idx,     monoData.length - 1)];
        const b    = monoData[Math.min(idx + 1, monoData.length - 1)];
        audioData[i] = a + frac * (b - a);
      }
    }
    console.log(`[WAV] dataOffset=${dataOffset} sr=${fileSr} bits=${bitsPerSample} ch=${numChannels} samples=${audioData.length}`);

    setStatus('Classifying...');
    const totalWindows = Math.max(0, Math.floor((audioData.length - WIN_SAMPLES) / STRIDE_SAMPLES) + 1);
    setProgress(0);
    const windowPredictions = [];
    let firstFeatures = null;
    let windowIdx = 0;
    for (let start = 0; start + WIN_SAMPLES <= audioData.length; start += STRIDE_SAMPLES) {
      const segment = audioData.slice(start, start + WIN_SAMPLES);
      const features = extractFeatures(segment);
      if (!firstFeatures) {
        firstFeatures = features;
        const scaled = features.map((v, i) => (v - model.scaler_mean[i]) / model.scaler_scale[i]);
        const kernelVals = model.support_vectors.map(sv => rbfKernel(scaled, sv, model.gamma));
        const maxK = Math.max(...kernelVals);
        const meanK = kernelVals.reduce((a, b) => a + b, 0) / kernelVals.length;
        console.log('[FEATURES] count:', features.length);
        console.log('[FEATURES] first 10 raw:', features.slice(0, 10));
        console.log('[FEATURES] first 10 scaled:', scaled.slice(0, 10));
        console.log('[KERNEL] max:', maxK, 'mean:', meanK);
        console.log('[KERNEL] non-zero count:', kernelVals.filter(v => v > 1e-10).length, '/', kernelVals.length);
      }
      windowPredictions.push(predict(features));
      windowIdx++;
      setProgress(windowIdx / totalWindows);
      // Yield to UI thread every 5 windows so the progress bar actually renders
      if (windowIdx % 5 === 0) await new Promise(r => setTimeout(r, 0));
    }
    setProgress(null);
    console.log('[PREDICT] window predictions:', windowPredictions);

    if (!windowPredictions.length) { setStatus('Audio too short to classify'); return; }
    const { label, confidence } = majorityVote(windowPredictions);
    setPrediction({ label, confidence });
    setStatus(`${windowPredictions.length} windows analysed`);
  };

  const uploadAudio = async () => {
    try {
      const result = await DocumentPicker.getDocumentAsync({
        type: 'audio/wav',        // WAV only — MP3 is compressed and can't be decoded directly
        copyToCacheDirectory: true,
      });
      if (result.canceled) return;
      const asset = result.assets[0];
      const name  = (asset.name ?? asset.uri ?? '').toLowerCase();
      if (!name.endsWith('.wav')) {
        setStatus('Only 16-bit WAV files are supported. Please convert to WAV first.');
        return;
      }
      setStatus('Processing uploaded file...');
      setPrediction(null);
      await processAudio(asset.uri);
    } catch (err) {
      setStatus(`Upload error: ${err.message}`);
      console.error('Failed to upload audio', err);
    }
  };

  return (
    <View style={styles.container}>
      <Text style={styles.title}>Frog Classifier</Text>
      <Button
        title={audioRecorder.isRecording ? 'Stop Recording' : 'Start Recording'}
        onPress={audioRecorder.isRecording ? stopRecording : startRecording}
        disabled={audioRecorder.isRecording ? false : false}
      />
      <Button title="Upload WAV File" onPress={uploadAudio} disabled={audioRecorder.isRecording} />
      <Text style={styles.status}>{status}</Text>
      {progress !== null ? (
        <View style={styles.progressTrack}>
          <View style={[styles.progressFill, { width: `${Math.round(progress * 100)}%` }]} />
        </View>
      ) : null}
      {prediction ? (
        <View style={styles.resultBox}>
          <Text style={[styles.prediction, prediction.label === 'Background' && styles.predictionBackground]}>
            {prediction.label === 'Background' ? 'No frog detected' : prediction.label}
          </Text>
          <Text style={styles.confidence}>
            {Math.round(prediction.confidence * 100)}% confidence
          </Text>
        </View>
      ) : null}
    </View>
  );
}

const styles = StyleSheet.create({
  container: { flex: 1, backgroundColor: '#fff', alignItems: 'center', justifyContent: 'center', gap: 16 },
  title: { fontSize: 24, fontWeight: 'bold' },
  status: { color: '#666', textAlign: 'center', paddingHorizontal: 16 },
  resultBox: { alignItems: 'center', gap: 4 },
  prediction: { fontSize: 20, fontWeight: '700', color: '#2a7' },
  predictionBackground: { color: '#999' },
  confidence: { fontSize: 14, color: '#666' },
  progressTrack: { width: '70%', height: 8, backgroundColor: '#e0e0e0', borderRadius: 4, overflow: 'hidden' },
  progressFill: { height: '100%', backgroundColor: '#2a7', borderRadius: 4 },
});

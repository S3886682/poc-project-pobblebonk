import { useState } from 'react';
import { View, Text, Button, StyleSheet } from 'react-native';
import { useAudioRecorder, AudioModule } from 'expo-audio';
import * as FileSystem from 'expo-file-system/legacy';
import { Buffer } from 'buffer';
import Meyda from 'meyda';

const model = require('./assets/svm_model.json');

// Match server.py parameters exactly
const SR = 32000;
const WIN_SAMPLES = Math.round(SR * 0.3);     // 9600
const STRIDE_SAMPLES = Math.round(SR * 0.2);  // 6400
const N_MFCC = 100;
const FFT_SIZE = 2048;
const HOP_SIZE = 512;
const WAV_HEADER_BYTES = 44;

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
  const decisions = model.intercept.map((inter, k) => {
    let sum = inter;
    for (let i = 0; i < model.support_vectors.length; i++) {
      sum += model.dual_coef[k][i] * rbfKernel(scaled, model.support_vectors[i], model.gamma);
    }
    return sum;
  });
  return model.classes[decisions.indexOf(Math.max(...decisions))];
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
function extractFeatures(segment) {
  Meyda.sampleRate = SR;
  Meyda.numberOfMFCCCoefficients = N_MFCC;

  const mfccFrames = [];
  const centroidFrames = [];
  for (let i = 0; i + FFT_SIZE <= segment.length; i += HOP_SIZE) {
    const frame = segment.slice(i, i + FFT_SIZE);
    const mfcc = Meyda.extract('mfcc', frame);
    const centroid = Meyda.extract('spectralCentroid', frame);
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
  const filtered = preds.filter(p => p !== 'Background');
  if (!filtered.length) return 'Background';
  const counts = {};
  for (const p of filtered) counts[p] = (counts[p] || 0) + 1;
  return Object.entries(counts).sort((a, b) => b[1] - a[1])[0][0];
}

// --- App ---
export default function App() {
  const audioRecorder = useAudioRecorder(RECORDING_OPTIONS);
  const [prediction, setPrediction] = useState('');
  const [status, setStatus] = useState('');

  const startRecording = async () => {
    try {
      setStatus('Requesting permission...');
      const { granted } = await AudioModule.requestRecordingPermissionsAsync();
      if (!granted) { setStatus('Microphone permission denied'); return; }
      setStatus('Starting...');
      await AudioModule.setAudioModeAsync({ allowsRecording: true, playsInSilentMode: true });
      await audioRecorder.prepareToRecordAsync();
      audioRecorder.record();
      setStatus('Recording');
    } catch (err) {
      setStatus(`Error: ${err.message}`);
      console.error('Failed to start recording', err);
    }
  };

  const stopRecording = async () => {
    try {
      setStatus('Processing...');
      const result = await audioRecorder.stop();
      let uri = result?.uri ?? audioRecorder.uri;

      if (!uri) {
        await new Promise(res => setTimeout(res, 300));
        uri = audioRecorder.uri;
      }

      if (!uri) { setStatus('Error: no recording URI after stop'); return; }
      await processAudio(uri);
    } catch (err) {
      setStatus(`Error: ${err.message}`);
      console.error('Failed to stop recording', err);
    }
  };

  const processAudio = async (uri) => {
    setStatus('Reading file...');
    const base64 = await FileSystem.readAsStringAsync(uri, { encoding: 'base64' });
    const bytes = Buffer.from(base64, 'base64');

    setStatus('Decoding PCM...');
    const nSamples = (bytes.length - WAV_HEADER_BYTES) / 2;
    const audioData = new Float32Array(nSamples);
    for (let i = 0; i < nSamples; i++) {
      const offset = WAV_HEADER_BYTES + i * 2;
      let s = bytes.readInt16LE(offset);
      audioData[i] = s / 32768;
    }

    setStatus('Classifying...');
    const windowPredictions = [];
    for (let start = 0; start + WIN_SAMPLES <= audioData.length; start += STRIDE_SAMPLES) {
      const segment = audioData.slice(start, start + WIN_SAMPLES);
      const features = extractFeatures(segment);
      windowPredictions.push(predict(features));
    }

    if (!windowPredictions.length) { setStatus('Recording too short'); return; }
    const result = majorityVote(windowPredictions);
    setPrediction(result);
    setStatus(`Done (${windowPredictions.length} windows)`);
  };

  return (
    <View style={styles.container}>
      <Text style={styles.title}>Frog Classifier</Text>
      <Button
        title={audioRecorder.isRecording ? 'Stop Recording' : 'Start Recording'}
        onPress={audioRecorder.isRecording ? stopRecording : startRecording}
      />
      <Text style={styles.status}>{status}</Text>
      {prediction ? <Text style={styles.prediction}>Prediction: {prediction}</Text> : null}
    </View>
  );
}

const styles = StyleSheet.create({
  container: { flex: 1, backgroundColor: '#fff', alignItems: 'center', justifyContent: 'center', gap: 16 },
  title: { fontSize: 24, fontWeight: 'bold' },
  status: { color: '#666' },
  prediction: { fontSize: 18, fontWeight: '600' },
});

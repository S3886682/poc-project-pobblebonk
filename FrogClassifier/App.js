import "./global.css";
import { useState, useEffect, useRef } from 'react';
import { View, Text, TouchableOpacity, Animated, StatusBar, StyleSheet, ScrollView, Dimensions } from 'react-native';
import { SafeAreaView, SafeAreaProvider } from 'react-native-safe-area-context';
import { useAudioRecorder, AudioModule } from 'expo-audio';
import { Ionicons } from '@expo/vector-icons';
import { BlurView } from 'expo-blur';
import * as Haptics from 'expo-haptics';
import { useFonts, Inter_400Regular, Inter_500Medium, Inter_600SemiBold, Inter_700Bold } from '@expo-google-fonts/inter';

import * as FileSystem from 'expo-file-system/legacy';
import * as DocumentPicker from 'expo-document-picker';
import { Buffer } from 'buffer';
import Meyda from 'meyda';

const model = require('./assets/svm_model.json');
const { width: SW, height: SH } = Dimensions.get('window');
const TRACK_W = SW - 40; // ScrollView paddingHorizontal 20 × 2

// Type scale
const T = {
  label:   { fontFamily: 'Inter-SemiBold', fontSize: 11, letterSpacing: 1.5 },
  caption: { fontFamily: 'Inter-Regular',  fontSize: 12 },
  body:    { fontFamily: 'Inter-Regular',  fontSize: 14, lineHeight: 20 },
  medium:  { fontFamily: 'Inter-Medium',   fontSize: 14 },
  title:   { fontFamily: 'Inter-SemiBold', fontSize: 15 },
  h3:      { fontFamily: 'Inter-Bold',     fontSize: 20 },
  h1:      { fontFamily: 'Inter-Bold',     fontSize: 40 },
};

const TEAM = [
  { name: 'Ashley', role: 'ML Engineer', initials: 'A' },
  { name: 'Jordan', role: 'Field Biologist', initials: 'J' },
  { name: 'Sam', role: 'iOS Developer', initials: 'S' },
];

const SIGHTINGS = [
  { species: 'Litoria caerulea',       location: 'Brisbane, QLD',   date: '18 Mar 2026', confidence: 94 },
  { species: 'Litoria fallax',          location: 'Gold Coast, QLD', date: '15 Mar 2026', confidence: 87 },
  { species: 'Crinia signifera',        location: 'Sydney, NSW',     date: '12 Mar 2026', confidence: 91 },
  { species: 'Uperoleia laevigata',     location: 'Darwin, NT',      date: '10 Mar 2026', confidence: 78 },
  { species: 'Limnodynastes peronii',   location: 'Melbourne, VIC',  date: '8 Mar 2026',  confidence: 85 },
];

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
  const filtered = preds.filter(p => p.label !== 'Background');
  if (!filtered.length) {
    return { label: 'Background', confidence: 1, all: [] };
  }
  const counts = {};
  for (const { label } of filtered) counts[label] = (counts[label] || 0) + 1;
  const sorted = Object.entries(counts).sort((a, b) => b[1] - a[1]);
  const winner = sorted[0][0];
  const all = sorted.map(([label, count]) => ({ label, confidence: count / filtered.length }));
  return { label: winner, confidence: all[0].confidence, all };
}

// --- App ---
export default function App() {
  const [fontsLoaded] = useFonts({
    'Inter-Regular':  Inter_400Regular,
    'Inter-Medium':   Inter_500Medium,
    'Inter-SemiBold': Inter_600SemiBold,
    'Inter-Bold':     Inter_700Bold,
  });

  const audioRecorder = useAudioRecorder(RECORDING_OPTIONS);
  const [prediction, setPrediction] = useState(null);
  const [status, setStatus] = useState('');
  const [progress, setProgress] = useState(null); // null = hidden, 0-1 = classifying

  // Pulse animation rings for recording state
  const pulse1 = useRef(new Animated.Value(1)).current;
  const pulse2 = useRef(new Animated.Value(1)).current;
  const opacity1 = useRef(new Animated.Value(0)).current;
  const opacity2 = useRef(new Animated.Value(0)).current;
  const loop1 = useRef(null);
  const loop2 = useRef(null);
  const cancelRef = useRef(false);
  const progressAnim = useRef(new Animated.Value(0)).current;
  const [isProcessing, setIsProcessing] = useState(false);
  const [activePage, setActivePage] = useState(0);
  const pageAnim    = useRef(new Animated.Value(0)).current;
  const tabPillAnim = useRef(new Animated.Value(0)).current;

  const goToPage = (page) => {
    setActivePage(page);
    Animated.timing(pageAnim,    { toValue: -page * SW, duration: 280, useNativeDriver: true }).start();
    Animated.spring(tabPillAnim, { toValue: page,       useNativeDriver: true, speed: 20, bounciness: 6 }).start();
  };

  useEffect(() => {
    if (audioRecorder.isRecording) {
      const makeLoop = (scale, opacity, delay) => {
        scale.setValue(1);
        opacity.setValue(0.6);
        return Animated.loop(
          Animated.sequence([
            Animated.delay(delay),
            Animated.parallel([
              Animated.timing(scale,   { toValue: 2.8, duration: 1600, useNativeDriver: true }),
              Animated.timing(opacity, { toValue: 0,   duration: 1600, useNativeDriver: true }),
            ]),
            Animated.parallel([
              Animated.timing(scale,   { toValue: 1,   duration: 0, useNativeDriver: true }),
              Animated.timing(opacity, { toValue: 0.6, duration: 0, useNativeDriver: true }),
            ]),
          ])
        );
      };
      loop1.current = makeLoop(pulse1, opacity1, 0);
      loop2.current = makeLoop(pulse2, opacity2, 700);
      loop1.current.start();
      loop2.current.start();
    } else {
      loop1.current?.stop();
      loop2.current?.stop();
      opacity1.setValue(0);
      opacity2.setValue(0);
      pulse1.setValue(1);
      pulse2.setValue(1);
    }
  }, [audioRecorder.isRecording]);

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
    }
  };

  const stopRecording = async () => {
    try {
      setStatus('Processing...');
      let uri;
      try {
        const result = await audioRecorder.stop();
        uri = result?.uri ?? audioRecorder.uri;
      } catch {
        uri = audioRecorder.uri;
      }

      if (!uri) {
        await new Promise(res => setTimeout(res, 300));
        uri = audioRecorder.uri;
      }

      if (!uri) { setStatus('Error: no recording URI after stop'); return; }
      await processAudio(uri);
    } catch (err) {
      setStatus(`Error: ${err.message}`);
    }
  };

  const processAudio = async (uri) => {
    setIsProcessing(true);
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
      setIsProcessing(false);
      setStatus('Error: could not parse WAV header');
      return;
    }
    if (bitsPerSample !== 16) {
      setIsProcessing(false);
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
    setStatus('Classifying...');
    const totalWindows = Math.max(0, Math.floor((audioData.length - WIN_SAMPLES) / STRIDE_SAMPLES) + 1);
    setProgress(0);
    progressAnim.setValue(0);
    cancelRef.current = false;
    const windowPredictions = [];
    let windowIdx = 0;
    for (let start = 0; start + WIN_SAMPLES <= audioData.length; start += STRIDE_SAMPLES) {
      if (cancelRef.current) break;
      const segment = audioData.slice(start, start + WIN_SAMPLES);
      windowPredictions.push(predict(extractFeatures(segment)));
      windowIdx++;
      const pct = windowIdx / totalWindows;
      setProgress(pct);
      Animated.timing(progressAnim, { toValue: pct, duration: 180, useNativeDriver: true }).start();
      await new Promise(r => setTimeout(r, 0)); // yield every window → smooth bar
    }
    setProgress(null);

    setIsProcessing(false);
    if (cancelRef.current) { setStatus('Cancelled'); return; }
    if (!windowPredictions.length) { setStatus('Audio too short to classify'); return; }
    setExpandedResults(false);
    const result = majorityVote(windowPredictions);
    setPrediction(result);
    setStatus(`${windowPredictions.length} windows analysed`);
    Haptics.notificationAsync(
      result.label !== 'Background'
        ? Haptics.NotificationFeedbackType.Success
        : Haptics.NotificationFeedbackType.Warning
    );
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
    }
  };

  const [expandedResults, setExpandedResults] = useState(false);

  if (!fontsLoaded) return null;

  const isFrog = prediction && prediction.label !== 'Background';
  const busy = isProcessing || audioRecorder.isRecording;

  return (
    <SafeAreaProvider>
      <SafeAreaView style={styles.safeArea}>
        <StatusBar barStyle="light-content" backgroundColor="#0a1f10" />

        {/* Sliding pages container */}
        <View style={{ flex: 1, overflow: 'hidden' }}>
        <Animated.View style={[styles.pagesRow, { transform: [{ translateX: pageAnim }] }]}>

          {/* ── PAGE 1: Classifier ── */}
          <View style={styles.page}>
            {/* Header */}
            <View style={{ paddingHorizontal: 24, paddingTop: 16, paddingBottom: 8 }}>
              <Text style={[T.label, { color: '#4ade80' }]}>AI POWERED</Text>
              <Text style={[T.h1, { color: 'white', marginTop: 4 }]}>FrogFinder</Text>
              <Text style={[T.body, { color: '#bbf7d0', marginTop: 4, opacity: 0.5 }]}>
                Point your mic at frog calls to identify species
              </Text>
            </View>

            {/* Hero — flex:1 keeps button centered regardless of bottom panel */}
            <View style={styles.hero}>
              {isProcessing ? (
                <View style={{ alignItems: 'center', gap: 16 }}>
                  <TouchableOpacity
                    onPress={() => { cancelRef.current = true; }}
                    activeOpacity={0.85}
                    style={[styles.recordBtn, styles.recordBtnCancel]}
                  >
                    <Ionicons name="close" size={44} color="white" />
                  </TouchableOpacity>
                  <Text style={[T.medium, { color: 'white', opacity: 0.6 }]}>Stop analysing</Text>
                </View>
              ) : (
                <View style={{ alignItems: 'center' }}>
                  <View style={styles.btnWrapper}>
                    <Animated.View style={[styles.ring, { transform: [{ scale: pulse1 }], opacity: opacity1 }]} />
                    <Animated.View style={[styles.ring, { transform: [{ scale: pulse2 }], opacity: opacity2 }]} />
                    <TouchableOpacity
                      onPress={() => {
                        Haptics.impactAsync(audioRecorder.isRecording
                          ? Haptics.ImpactFeedbackStyle.Light
                          : Haptics.ImpactFeedbackStyle.Medium);
                        audioRecorder.isRecording ? stopRecording() : startRecording();
                      }}
                      activeOpacity={0.85}
                      style={[styles.recordBtn, audioRecorder.isRecording ? styles.recordBtnActive : styles.recordBtnIdle]}
                    >
                      <Ionicons name={audioRecorder.isRecording ? 'stop' : 'mic'} size={44} color="white" />
                    </TouchableOpacity>
                  </View>
                  <Text style={[T.medium, { color: 'white', marginTop: 24, opacity: 0.6 }]}>
                    {audioRecorder.isRecording ? 'Listening — tap to stop' : 'Tap to record'}
                  </Text>
                </View>
              )}
            </View>

            {/* Bottom panel — absolute so it never affects the hero's centering */}
            <View style={styles.bottomPanel}>
              {/* Scrollable dynamic content — maxHeight ensures ScrollView actually scrolls */}
              <ScrollView
                style={{ maxHeight: SH * 0.38 }}
                contentContainerStyle={{ paddingHorizontal: 20, paddingTop: 12, paddingBottom: 8, gap: 10 }}
                showsVerticalScrollIndicator={false}
              >
                {progress !== null && (
                  <View>
                    <Text style={[T.caption, { color: '#86efac', opacity: 0.7, marginBottom: 8 }]}>
                      Analysing — {Math.round(progress * 100)}%
                    </Text>
                    <View style={styles.confTrack}>
                      <Animated.View style={[styles.confFill, {
                        width: TRACK_W,
                        transform: [{ translateX: progressAnim.interpolate({ inputRange: [0, 1], outputRange: [-TRACK_W, 0] }) }],
                      }]} />
                    </View>
                  </View>
                )}

                {prediction && (
                  <BlurView intensity={28} tint="dark" style={styles.resultCard}>
                    <Text style={[T.label, { color: '#4ade80', marginBottom: 12 }]}>
                      {isFrog ? 'DETECTED SPECIES' : 'RESULT'}
                    </Text>

                    {isFrog ? (
                      <>
                        {(() => {
                          const top = prediction.all[0];
                          const pct = Math.round(top.confidence * 100);
                          return (
                            <View>
                              <View style={{ flexDirection: 'row', justifyContent: 'space-between', marginBottom: 6 }}>
                                <Text style={[T.title, { color: 'white' }]}>{top.label}</Text>
                                <Text style={[T.title, { color: '#4ade80' }]}>{pct}%</Text>
                              </View>
                              <View style={styles.confTrack}>
                                <View style={[styles.confFill, { width: `${pct}%` }]} />
                              </View>
                            </View>
                          );
                        })()}

                        {expandedResults && (
                          <ScrollView style={{ maxHeight: 160, marginTop: 4 }} nestedScrollEnabled showsVerticalScrollIndicator={false}>
                            {prediction.all.slice(1).map((item) => {
                              const pct = Math.round(item.confidence * 100);
                              return (
                                <View key={item.label} style={{ marginTop: 12, opacity: 0.45 }}>
                                  <View style={{ flexDirection: 'row', justifyContent: 'space-between', marginBottom: 6 }}>
                                    <Text style={[T.medium, { color: 'white' }]}>{item.label}</Text>
                                    <Text style={[T.medium, { color: '#4ade80' }]}>{pct}%</Text>
                                  </View>
                                  <View style={styles.confTrack}>
                                    <View style={[styles.confFill, { width: `${pct}%` }]} />
                                  </View>
                                </View>
                              );
                            })}
                          </ScrollView>
                        )}

                        {prediction.all.length > 1 && (
                          <TouchableOpacity onPress={() => setExpandedResults(e => !e)} activeOpacity={0.7} style={styles.expandBtn}>
                            <Text style={[T.caption, { color: 'rgba(255,255,255,0.4)' }]}>
                              {expandedResults ? 'Hide others' : `Also detected · ${prediction.all.length - 1} other species`}
                            </Text>
                            <Ionicons name={expandedResults ? 'chevron-up' : 'chevron-down'} size={14} color="rgba(255,255,255,0.4)" />
                          </TouchableOpacity>
                        )}
                      </>
                    ) : (
                      <Text style={[T.h3, { color: 'white', opacity: 0.3 }]}>No frog detected</Text>
                    )}
                  </BlurView>
                )}

                {status ? (
                  <Text style={[T.caption, { color: 'white', opacity: 0.3, textAlign: 'center' }]}>{status}</Text>
                ) : null}
              </ScrollView>

              {/* Upload — always pinned at the bottom of the panel */}
              <View style={styles.uploadRow}>
                <TouchableOpacity
                  onPress={uploadAudio}
                  disabled={busy}
                  activeOpacity={0.7}
                  style={[styles.uploadBtn, busy && styles.uploadBtnDisabled]}
                >
                  <Ionicons name="cloud-upload-outline" size={17} color="rgba(255,255,255,0.6)" />
                  <Text style={[T.medium, { color: 'white', opacity: 0.6, marginLeft: 8 }]}>Upload WAV File</Text>
                </TouchableOpacity>
              </View>
            </View>
          </View>

          {/* ── PAGE 2: About ── */}
          <View style={styles.page}>
            <View style={{ paddingHorizontal: 24, paddingTop: 16, paddingBottom: 8 }}>
              <Text style={[T.label, { color: '#4ade80' }]}>LEARN MORE</Text>
              <Text style={[T.h1, { color: 'white', marginTop: 4 }]}>About</Text>
            </View>

            <ScrollView style={{ flex: 1 }} showsVerticalScrollIndicator={false} contentContainerStyle={{ padding: 24, gap: 16 }}>
              {/* Mission */}
              <View style={styles.aboutCard}>
                <Text style={[T.label, { color: '#4ade80', marginBottom: 8 }]}>OUR MISSION</Text>
                <Text style={[T.body, { color: 'white', opacity: 0.65 }]}>
                  FrogFinder uses on-device machine learning to identify frog species from their calls — no internet required.
                  By crowdsourcing detections, we help researchers track frog populations and detect early signs of habitat stress.
                </Text>
              </View>

              {/* Team */}
              <View style={styles.aboutCard}>
                <Text style={[T.label, { color: '#4ade80', marginBottom: 12 }]}>THE TEAM</Text>
                {TEAM.map((member) => (
                  <View key={member.name} style={{ flexDirection: 'row', alignItems: 'center', marginBottom: 12 }}>
                    <View style={styles.avatar}>
                      <Text style={[T.title, { color: 'white' }]}>{member.initials}</Text>
                    </View>
                    <View style={{ marginLeft: 12 }}>
                      <Text style={[T.title, { color: 'white' }]}>{member.name}</Text>
                      <Text style={[T.caption, { color: 'white', opacity: 0.4 }]}>{member.role}</Text>
                    </View>
                  </View>
                ))}
              </View>

              {/* Sightings */}
              <View style={styles.aboutCard}>
                <Text style={[T.label, { color: '#4ade80', marginBottom: 12 }]}>RECENT SIGHTINGS</Text>
                {SIGHTINGS.map((s, i) => (
                  <View key={i} style={[{ paddingBottom: 12 }, i < SIGHTINGS.length - 1 && styles.sightingDivider]}>
                    <View style={{ flexDirection: 'row', justifyContent: 'space-between', alignItems: 'center' }}>
                      <Text style={[T.medium, { color: 'white', flex: 1 }]}>{s.species}</Text>
                      <View style={styles.badge}>
                        <Text style={[T.label, { color: '#4ade80' }]}>{s.confidence}%</Text>
                      </View>
                    </View>
                    <Text style={[T.caption, { color: 'white', opacity: 0.4, marginTop: 3 }]}>
                      {s.location} · {s.date}
                    </Text>
                  </View>
                ))}
              </View>
            </ScrollView>
          </View>
        </Animated.View>
        </View>

        {/* Tab bar */}
        <View style={styles.tabBar}>
          <Animated.View style={[styles.tabPill, {
            transform: [{ translateX: tabPillAnim.interpolate({ inputRange: [0, 1], outputRange: [4, SW / 2 + 4] }) }],
          }]} />
          {[
            { icon: 'mic-outline',                label: 'Identify', page: 0 },
            { icon: 'information-circle-outline', label: 'About',    page: 1 },
          ].map(({ icon, label, page }) => {
            const active = activePage === page;
            return (
              <TouchableOpacity
                key={page}
                onPress={() => goToPage(page)}
                disabled={busy}
                style={styles.tabItem}
                activeOpacity={0.7}
              >
                <Ionicons name={icon} size={22} color={active ? '#4ade80' : 'rgba(255,255,255,0.35)'} />
                <Text style={[T.caption, {
                  fontFamily: active ? 'Inter-SemiBold' : 'Inter-Regular',
                  fontSize: 10,
                  color: active ? '#4ade80' : 'rgba(255,255,255,0.35)',
                  marginTop: 3,
                }]}>
                  {label}
                </Text>
              </TouchableOpacity>
            );
          })}
        </View>
      </SafeAreaView>
    </SafeAreaProvider>
  );
}

const styles = StyleSheet.create({
  safeArea:     { flex: 1, backgroundColor: '#0a1f10' },
  pagesRow:     { position: 'absolute', top: 0, bottom: 0, left: 0, flexDirection: 'row', width: SW * 2 },
  page:         { width: SW, height: '100%' },
  hero:         { flex: 1, alignItems: 'center', justifyContent: 'center', paddingBottom: 80 },
  bottomPanel:  { position: 'absolute', bottom: 0, left: 0, right: 0, maxHeight: '52%' },
  btnWrapper:   { width: 108, height: 108, alignItems: 'center', justifyContent: 'center' },
  ring: {
    position: 'absolute', width: 108, height: 108, borderRadius: 54,
    borderWidth: 2, borderColor: '#4ade80',
  },
  recordBtn: {
    width: 108, height: 108, borderRadius: 54,
    alignItems: 'center', justifyContent: 'center',
    shadowOffset: { width: 0, height: 0 }, shadowRadius: 24, shadowOpacity: 0.6, elevation: 12,
  },
  recordBtnCancel: { backgroundColor: '#6b7280', shadowColor: '#9ca3af' },
  recordBtnIdle:   { backgroundColor: '#16a34a', shadowColor: '#4ade80' },
  recordBtnActive: { backgroundColor: '#dc2626', shadowColor: '#f87171' },
  resultCard: {
    borderWidth: 1, borderColor: 'rgba(255,255,255,0.12)',
    borderRadius: 20, padding: 20, overflow: 'hidden',
  },
  confTrack: { height: 4, backgroundColor: 'rgba(255,255,255,0.1)', borderRadius: 99, overflow: 'hidden' },
  confFill:  { height: '100%', backgroundColor: '#4ade80', borderRadius: 99 },
  uploadRow: {
    paddingHorizontal: 20, paddingBottom: 16, paddingTop: 8,
    borderTopWidth: 1, borderTopColor: 'rgba(255,255,255,0.06)',
    backgroundColor: '#0a1f10',
  },
  uploadBtn: {
    flexDirection: 'row', alignItems: 'center', justifyContent: 'center',
    paddingVertical: 14, borderRadius: 14,
    borderWidth: 1, borderColor: 'rgba(255,255,255,0.15)',
  },
  uploadBtnDisabled: { opacity: 0.3 },
  expandBtn: {
    flexDirection: 'row', alignItems: 'center', justifyContent: 'center',
    gap: 4, marginTop: 14, paddingTop: 12,
    borderTopWidth: 1, borderTopColor: 'rgba(255,255,255,0.08)',
  },
  tabBar: {
    flexDirection: 'row', borderTopWidth: 1,
    borderTopColor: 'rgba(255,255,255,0.08)',
    backgroundColor: '#0a1f10',
  },
  tabItem:  { flex: 1, alignItems: 'center', justifyContent: 'center', paddingVertical: 10, zIndex: 1 },
  tabPill:  {
    position: 'absolute', top: 6, bottom: 6,
    width: SW / 2 - 8, borderRadius: 10,
    backgroundColor: 'rgba(74,222,128,0.12)',
  },
  aboutCard: {
    backgroundColor: 'rgba(255,255,255,0.06)',
    borderWidth: 1, borderColor: 'rgba(255,255,255,0.1)',
    borderRadius: 18, padding: 18,
  },
  avatar: {
    width: 40, height: 40, borderRadius: 20,
    backgroundColor: 'rgba(74,222,128,0.2)',
    alignItems: 'center', justifyContent: 'center',
  },
  badge: {
    backgroundColor: 'rgba(74,222,128,0.15)',
    borderRadius: 8, paddingHorizontal: 8, paddingVertical: 3,
  },
  sightingDivider: { borderBottomWidth: 1, borderBottomColor: 'rgba(255,255,255,0.08)', marginBottom: 12 },
});

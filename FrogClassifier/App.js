import "./global.css";
import { useState, useEffect, useRef } from 'react';
import { View, Text, TouchableOpacity, Animated, Easing, StatusBar, StyleSheet, ScrollView, Dimensions } from 'react-native';
import { SafeAreaView, SafeAreaProvider, useSafeAreaInsets } from 'react-native-safe-area-context';
import { useAudioRecorder, AudioModule } from 'expo-audio';
import { Ionicons } from '@expo/vector-icons';
import * as Haptics from 'expo-haptics';
import { useFonts, Nunito_400Regular, Nunito_600SemiBold, Nunito_700Bold, Nunito_800ExtraBold } from '@expo-google-fonts/nunito';

import * as FileSystem from 'expo-file-system/legacy';
import * as DocumentPicker from 'expo-document-picker';
import { Buffer } from 'buffer';
import Meyda from 'meyda';

const model = require('./assets/svm_model.json');
const { width: SW, height: SH } = Dimensions.get('window');
const TRACK_W = SW - 40; // ScrollView paddingHorizontal 20 × 2

// Palette — pastel AC tones
const C = {
  bg:         '#CCE9F5',  // soft pastel sky
  surface:    '#FEFBF4',  // warm cream cards
  white:      '#FFFFFF',
  border:     'rgba(172,144,98,0.22)',
  green:      '#72B55F',  // pastel sage
  greenLight: '#DFF2E4',
  mint:       '#D5EFE0',
  yellow:     '#FFE08A',  // pastel butter yellow
  yellowDark: '#C49225',
  brown:      '#3D3226',
  brownMid:   '#8B7355',
  brownLight: '#C8AE88',
  tabBg:      '#EDE7DA',
  coral:      '#F09A87',  // pastel coral
  redLight:   '#FFE0DC',  // pastel red for no-frogs header
};

// Type scale
const T = {
  label:   { fontFamily: 'Nunito-Bold',       fontSize: 11, letterSpacing: 1.2 },
  caption: { fontFamily: 'Nunito-Regular',    fontSize: 13 },
  body:    { fontFamily: 'Nunito-Regular',    fontSize: 14, lineHeight: 22 },
  medium:  { fontFamily: 'Nunito-SemiBold',   fontSize: 14 },
  title:   { fontFamily: 'Nunito-Bold',       fontSize: 15 },
  h3:      { fontFamily: 'Nunito-ExtraBold',  fontSize: 20 },
  h1:      { fontFamily: 'Nunito-ExtraBold',  fontSize: 38 },
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

// Common name → latin name for all model classes
const LATIN_NAMES = {
  'Australian Lace-Lid':            'Nyctimystes dayi',
  'Baw Baw Frog':                   'Philoria frosti',
  'Beautiful Nursery Frog':         'Cophixalus concinnus',
  'Bellenden Ker Nursery Frog':     'Cophixalus neglectus',
  'Booroolong Frog':                'Litoria booroolongensis',
  'Cave Frog':                      'Litoria cavernicola',
  'Common Eastern Froglet':         'Crinia signifera',
  'Davies Tree Frog':               'Litoria daviesae',
  'Desert Spadefoot':               'Notaden nichollsi',
  'Eastern Banjo Frog':             'Limnodynastes dumerilii',
  'Eungella Day Frog':              'Taudactylus eungellensis',
  'Flat Headed Frog':               'Litoria inermis',
  "Fleay's Barred Frog":            'Mixophyes fleayi',
  'Giant Burrowing Frog':           'Heleioporus australiacus',
  'Green Tree Frog':                'Litoria caerulea',
  'Green and Golden Bell Frog':     'Litoria aurea',
  "Hosmer's Nursery Frog":          'Cophixalus hosmeri',
  'Howard Springs Toadlet':         'Uperoleia inundata',
  'Kroombit Tops Tinker Frog':      'Taudactylus pleione',
  'Kuranda Tree Frog':              'Litoria myola',
  "Littlejohn's Toadlet":           'Uperoleia littlejohni',
  'Magnificent Brood Frog':         'Pseudophryne covacevichae',
  'Magnificent Tree Frog':          'Litoria splendida',
  "Mahony's Toadlet":               'Uperoleia mahonyi',
  'Moss Froglet':                   'Bryobatrachus nimbus',
  'Motorbike Frog':                 'Litoria moorei',
  'Mount Top Nursery Frog':         'Cophixalus monticola',
  'Mountain Frog':                  'Philoria sphagnicolus',
  'Mountain Mist Frog':             'Litoria nyakalensis',
  'Mt Elliot Nursery Frog':         'Cophixalus mcdonaldi',
  'Northern Corroboree Frog':       'Pseudophryne pengilleyi',
  'Northern Flinders Ranges Froglet': 'Crinia flindersensis',
  'Northern Heath Frog':            'Litoria olongburensis',
  'Northern Snapping Frog':         'Cyclorana australis',
  'Northern Tinker Frog':           'Taudactylus rheophilus',
  'Orange-bellied Froglet':         'Crinia parinsignifera',
  'Pobblebonk':                     'Limnodynastes dumerilii',
  'Rattling Nursery Frog':          'Cophixalus rattus',
  'Southern Barred Frog':           'Mixophyes balbus',
  'Southern Bell Frog':             'Litoria raniformis',
  'Southern Corroboree Frog':       'Pseudophryne corroboree',
  'Southern Heath Frog':            'Litoria heinrichii',
  'Spotted Tree Frog':              'Litoria spenceri',
  'Sunset Frog':                    'Spicospina flammocaerulea',
  'Tapping Nursery Frog':           'Cophixalus bombiens',
  'Tasmanian Tree Frog':            'Litoria burrowsae',
  'Tusked Frog':                    'Adelotus brevis',
  'Victorian Smooth Froglet':       'Geocrinia victoriana',
  'Wallum Sedge Frog':              'Litoria olongburensis',
  'White Bellied Frog':             'Geocrinia alba',
  'Yellow Spotted Bell Frog':       'Litoria castanea',
};
// Model outputs common names directly
const getNames = (label) => ({ displayName: label, latin: LATIN_NAMES[label] ?? null });

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

// Flat AC-style cloud shape
const FlatCloud = ({ width = 100, opacity = 0.88 }) => {
  const b = width * 0.18; // bump radius scales with cloud width
  return (
    <View style={{ width, height: b * 3.2, opacity }}>
      {/* Flat base */}
      <View style={{ position: 'absolute', bottom: 0, left: 0, right: 0, height: b * 1.1, backgroundColor: 'white', borderRadius: b * 0.55 }} />
      {/* Three bumps, naturally overlapping */}
      <View style={{ position: 'absolute', bottom: b * 0.55, left: width * 0.08, width: b * 2, height: b * 2, backgroundColor: 'white', borderRadius: 999 }} />
      <View style={{ position: 'absolute', bottom: b * 0.9,  left: width * 0.30, width: b * 2.6, height: b * 2.6, backgroundColor: 'white', borderRadius: 999 }} />
      <View style={{ position: 'absolute', bottom: b * 0.55, right: width * 0.10, width: b * 1.7, height: b * 1.7, backgroundColor: 'white', borderRadius: 999 }} />
    </View>
  );
};

// --- App ---
function AppContent() {
  const insets = useSafeAreaInsets();
  const [fontsLoaded] = useFonts({
    'Nunito-Regular':   Nunito_400Regular,
    'Nunito-SemiBold':  Nunito_600SemiBold,
    'Nunito-Bold':      Nunito_700Bold,
    'Nunito-ExtraBold': Nunito_800ExtraBold,
  });

  const audioRecorder = useAudioRecorder(RECORDING_OPTIONS);
  const [prediction, setPrediction] = useState(null);
  const [status, setStatus] = useState('');
  const [progress, setProgress] = useState(null);
  const [audioDuration, setAudioDuration] = useState(0);

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
  const pressAnim   = useRef(new Animated.Value(0)).current;
  const cloud1      = useRef(new Animated.Value(SW + 80)).current;
  const cloud2      = useRef(new Animated.Value(SW + 80)).current;
  const cloud3      = useRef(new Animated.Value(SW + 80)).current;
  const shakeAnim       = useRef(new Animated.Value(0)).current;
  const resultDropAnim  = useRef(new Animated.Value(1)).current;
  const progressFadeAnim = useRef(new Animated.Value(0)).current;
  const heroCrossAnim   = useRef(new Animated.Value(1)).current;
  const isMounted      = useRef(false);
  const [expandedResults, setExpandedResults] = useState(false);

  const pressIn  = () => Animated.spring(pressAnim, { toValue: 4, useNativeDriver: true, speed: 60, bounciness: 0 }).start();
  const pressOut = () => Animated.spring(pressAnim, { toValue: 0, useNativeDriver: true, speed: 20, bounciness: 6 }).start();

  const goToPage = (page) => {
    setActivePage(page);
    Animated.timing(pageAnim,    { toValue: -page * SW, duration: 280, useNativeDriver: true }).start();
    Animated.spring(tabPillAnim, { toValue: page, useNativeDriver: true, speed: 14, bounciness: 12 }).start();
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

  // Drifting clouds
  useEffect(() => {
    const startCloud = (anim, duration, delay) => {
      anim.setValue(SW + 80);
      Animated.loop(
        Animated.sequence([
          Animated.delay(delay),
          Animated.timing(anim, { toValue: -160, duration, useNativeDriver: true }),
          Animated.timing(anim, { toValue: SW + 80, duration: 0, useNativeDriver: true }),
        ])
      ).start();
    };
    startCloud(cloud1, 20000,     0);
    startCloud(cloud2, 28000,  8000);
    startCloud(cloud3, 16000, 14000);
  }, []);

  // Crossfade hero area on state transitions
  useEffect(() => {
    if (!isMounted.current) { isMounted.current = true; return; }
    heroCrossAnim.setValue(0.15);
    Animated.timing(heroCrossAnim, { toValue: 1, duration: 280, useNativeDriver: true }).start();
  }, [isProcessing, audioRecorder.isRecording]);

  // Periodic frog shake
  useEffect(() => {
    let timeout;
    const scheduleShake = () => {
      timeout = setTimeout(() => {
        Animated.sequence([
          Animated.timing(shakeAnim, { toValue: -9, duration: 55, useNativeDriver: true }),
          Animated.timing(shakeAnim, { toValue:  9, duration: 55, useNativeDriver: true }),
          Animated.timing(shakeAnim, { toValue: -6, duration: 55, useNativeDriver: true }),
          Animated.timing(shakeAnim, { toValue:  6, duration: 55, useNativeDriver: true }),
          Animated.timing(shakeAnim, { toValue:  0, duration: 55, useNativeDriver: true }),
        ]).start(() => scheduleShake());
      }, 3500 + Math.random() * 3000);
    };
    scheduleShake();
    return () => clearTimeout(timeout);
  }, []);

  const dropAndClearPrediction = (cb) => {
    if (!prediction) { cb?.(); return; }
    Animated.timing(resultDropAnim, { toValue: 0, duration: 200, useNativeDriver: true }).start(() => {
      setPrediction(null);
      cb?.();
    });
  };

  const startRecording = async () => {
    dropAndClearPrediction(async () => {
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
    });
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
    progressFadeAnim.setValue(0);
    setProgress(0);
    progressAnim.setValue(0);
    cancelRef.current = false;
    Animated.timing(progressFadeAnim, { toValue: 1, duration: 220, useNativeDriver: true }).start();

    // Benchmark one window to estimate total duration, then run one smooth animation
    // decoupled from the processing loop so the bar never jumps.
    const benchStart = Date.now();
    predict(extractFeatures(audioData.slice(0, WIN_SAMPLES)));
    const estimatedMs = Math.max(1500, (Date.now() - benchStart) * totalWindows * 1.3);
    const progAnim = Animated.timing(progressAnim, {
      toValue: 0.92,
      duration: estimatedMs,
      easing: Easing.out(Easing.cubic),
      useNativeDriver: true,
    });
    progAnim.start();

    const windowPredictions = [];
    let windowIdx = 0;
    for (let start = 0; start + WIN_SAMPLES <= audioData.length; start += STRIDE_SAMPLES) {
      if (cancelRef.current) break;
      const segment = audioData.slice(start, start + WIN_SAMPLES);
      windowPredictions.push(predict(extractFeatures(segment)));
      windowIdx++;
      setProgress(windowIdx / totalWindows);
      await new Promise(r => setTimeout(r, 0)); // keep UI responsive
    }

    // Stop the independent animation and snap to finish
    progAnim.stop();
    await new Promise(r =>
      Animated.timing(progressAnim, { toValue: 1, duration: 300, easing: Easing.out(Easing.quad), useNativeDriver: true }).start(r)
    );
    // Fade out the progress bar before showing result
    await new Promise(r =>
      Animated.timing(progressFadeAnim, { toValue: 0, duration: 240, useNativeDriver: true }).start(r)
    );
    setProgress(null);

    setIsProcessing(false);
    if (cancelRef.current) { setStatus('Cancelled'); return; }
    if (!windowPredictions.length) { setStatus('Audio too short to classify'); return; }
    setAudioDuration(Math.round(audioData.length / SR));
    const result = majorityVote(windowPredictions);
    setExpandedResults(false);
    resultDropAnim.setValue(0);
    setPrediction(result);
    Animated.timing(resultDropAnim, { toValue: 1, duration: 320, useNativeDriver: true }).start();
    setStatus('');
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


  if (!fontsLoaded) return null;

  const isFrog = prediction && prediction.label !== 'Background';
  const busy = isProcessing || audioRecorder.isRecording;

  return (
    <SafeAreaView style={styles.safeArea} edges={['top', 'left', 'right']}>
        <StatusBar barStyle="dark-content" backgroundColor={C.bg} />

        {/* Sliding pages container */}
        <View style={{ flex: 1, overflow: 'hidden' }}>
        <Animated.View style={[styles.pagesRow, { transform: [{ translateX: pageAnim }] }]}>

          {/* ── PAGE 1: Classifier ── */}
          <View style={styles.page}>
            {/* Drifting flat clouds */}
            <Animated.View style={[styles.cloud, { top: SH * 0.09, transform: [{ translateX: cloud1 }] }]}><FlatCloud width={120} opacity={0.85} /></Animated.View>
            <Animated.View style={[styles.cloud, { top: SH * 0.18, transform: [{ translateX: cloud2 }] }]}><FlatCloud width={80}  opacity={0.70} /></Animated.View>
            <Animated.View style={[styles.cloud, { top: SH * 0.04, transform: [{ translateX: cloud3 }] }]}><FlatCloud width={100} opacity={0.78} /></Animated.View>
            {/* Header */}
            <View style={{ paddingHorizontal: 24, paddingTop: 8, paddingBottom: 10 }}>
              {/* Mascot — absolutely positioned, no layout impact */}
              <View style={{ position: 'absolute', top: 8, right: 24, zIndex: 2 }}>
                <View style={styles.mascotCircle}>
                  <Text style={{ fontSize: 36 }}>🐸</Text>
                </View>
              </View>
              {/* Badge */}
              <View style={{ alignSelf: 'flex-start' }}>
                <View style={[styles.fieldGuideBadge, { position: 'absolute', top: 4, backgroundColor: '#b89a10' }]} />
                <View style={styles.fieldGuideBadge}>
                  <Text style={{ fontSize: 14 }}>🐸</Text>
                  <Text style={[T.label, { color: C.brown, marginLeft: 6 }]}>YOUR FIELD GUIDE</Text>
                </View>
              </View>
              <Text style={[T.h1, { color: C.brown, marginTop: 3, paddingRight: 88, textShadowColor: 'rgba(61,50,38,0.22)', textShadowOffset: { width: 2, height: 4 }, textShadowRadius: 1 }]}>FrogFinder</Text>
              {/* Description box — Ribbit label overlaps the top-left */}
              <View style={{ marginTop: 14 }}>
                <View style={styles.ribbitLabel}>
                  <Text style={[T.label, { color: C.brown }]}>Ribbit! 🐸</Text>
                </View>
                <View style={styles.descriptionBox}>
                  <Text style={[T.body, { color: C.brownMid }]}>
                    Point your phone at frog calls to discover who's singing!
                  </Text>
                </View>
              </View>
            </View>

            {/* AC dialogue — floats between header and button */}
            {prediction && (
              <Animated.View style={{ paddingHorizontal: 20, paddingBottom: 8, opacity: resultDropAnim }}>
                <View style={[styles.acDialogue, { position: 'absolute', bottom: 3, left: 20, right: 20, backgroundColor: C.brownLight, zIndex: 0 }]} />
                <View style={[styles.acDialogue, { zIndex: 1 }]}>
                  <Text style={[T.body, { color: C.brown, fontSize: 15, lineHeight: 24 }]}>
                    {isFrog
                      ? <>{'Wowie!! I found a friend out there~ 🌿\nIt sounds like a '}<Text style={{ fontFamily: 'Nunito-ExtraBold', color: C.green }}>{getNames(prediction.label).displayName}</Text>{'!'}</>
                      : "Hmm... I couldn't hear any frogs nearby! 🤔\nTry getting a little closer~"
                    }
                  </Text>
                </View>
              </Animated.View>
            )}

            {/* Hero — flex:1, crossfades on state transitions */}
            <Animated.View style={[styles.hero, { opacity: heroCrossAnim }]}>
              {isProcessing ? (
                <View style={{ alignItems: 'center', gap: 16 }}>
                  <View style={{ position: 'relative' }}>
                    <View style={[styles.recordBtn, styles.recordBtnShadowCancel, { position: 'absolute', top: 4 }]} />
                    <Animated.View style={{ transform: [{ translateY: pressAnim }] }}>
                      <TouchableOpacity
                        onPressIn={pressIn} onPressOut={pressOut}
                        onPress={() => { cancelRef.current = true; }}
                        activeOpacity={1}
                        style={[styles.recordBtn, styles.recordBtnCancel]}
                      >
                        <Ionicons name="close" size={40} color={C.brown} />
                      </TouchableOpacity>
                    </Animated.View>
                  </View>
                  <Text style={[T.medium, { color: C.brownMid }]}>Stop listening</Text>
                </View>
              ) : (
                <View style={{ alignItems: 'center' }}>
                  <View style={styles.btnWrapper}>
                    <Animated.View style={[styles.ring, { transform: [{ scale: pulse1 }], opacity: opacity1 }]} />
                    <Animated.View style={[styles.ring, { transform: [{ scale: pulse2 }], opacity: opacity2 }]} />
                    <View style={{ position: 'relative' }}>
                      <View style={[styles.recordBtn,
                        audioRecorder.isRecording ? styles.recordBtnShadowActive : styles.recordBtnShadowIdle,
                        { position: 'absolute', top: 4 }
                      ]} />
                      <Animated.View style={{ transform: [{ translateY: pressAnim }] }}>
                        <TouchableOpacity
                          onPressIn={pressIn} onPressOut={pressOut}
                          onPress={() => {
                            Haptics.impactAsync(Haptics.ImpactFeedbackStyle.Rigid);
                            audioRecorder.isRecording ? stopRecording() : startRecording();
                          }}
                          activeOpacity={1}
                          style={[styles.recordBtn, audioRecorder.isRecording ? styles.recordBtnActive : styles.recordBtnIdle]}
                        >
                          <Ionicons name={audioRecorder.isRecording ? 'stop' : 'mic'} size={40} color="white" />
                        </TouchableOpacity>
                      </Animated.View>
                    </View>
                  </View>
                  <Text style={[T.title, { color: C.brownMid, marginTop: 24 }]}>
                    {audioRecorder.isRecording ? 'Listening for frogs...' : 'Tap to listen!'}
                  </Text>
                </View>
              )}
            </Animated.View>

            {/* Bottom panel */}
            <View style={styles.bottomPanel}>
              {/* Scrollable dynamic content — maxHeight ensures ScrollView actually scrolls */}
              <ScrollView
                style={{ maxHeight: SH * 0.33 }}
                contentContainerStyle={{ paddingHorizontal: 20, paddingTop: 12, paddingBottom: 8, gap: 10 }}
                showsVerticalScrollIndicator={false}
              >
                {progress !== null && (
                  <Animated.View style={{ opacity: progressFadeAnim }}>
                    <Text style={[T.caption, { color: C.brownMid, marginBottom: 8 }]}>
                      Searching... {Math.round(progress * 100)}%
                    </Text>
                    <View style={styles.confTrack}>
                      <Animated.View style={[styles.confFill, {
                        width: TRACK_W,
                        transform: [{ translateX: progressAnim.interpolate({ inputRange: [0, 1], outputRange: [-TRACK_W, 0] }) }],
                      }]} />
                    </View>
                  </Animated.View>
                )}

                {prediction && (
                  <Animated.View style={{ opacity: resultDropAnim }}>
                    <View style={styles.resultCard}>
                      {/* Header strip — green for frog found, light red for no frogs */}
                      <View style={[styles.resultCardHeader, !isFrog && { backgroundColor: C.redLight }]}>
                        <Text style={[T.title, { color: C.green }]}>
                          {isFrog ? '⭐  FOUND A FRIEND!' : '🤔  NO FROGS NEARBY'}
                        </Text>
                        {audioDuration > 0 && (
                          <View style={styles.durationPill}>
                            <Text style={[T.caption, { color: C.brownMid }]}>{audioDuration}s analysed</Text>
                          </View>
                        )}
                      </View>

                      {/* Body — only render when there's something to show */}
                      {isFrog && (
                        <View style={{ padding: 16 }}>
                          {/* Winner row */}
                          {(() => {
                            const top = prediction.all[0];
                            const pct = Math.round(top.confidence * 100);
                            const { displayName, latin } = getNames(top.label);
                            return (
                              <View style={{ flexDirection: 'row', alignItems: 'center', gap: 14 }}>
                                <Animated.View style={{ transform: [{ rotate: shakeAnim.interpolate({ inputRange: [-9, 0, 9], outputRange: ['-12deg', '0deg', '12deg'] }) }] }}>
                                  <View style={styles.frogSquareLarge}>
                                    <Text style={{ fontSize: 40 }}>🐸</Text>
                                  </View>
                                </Animated.View>
                                <View style={{ flex: 1 }}>
                                  <Text style={[T.h3, { color: C.brown }]}>{displayName}</Text>
                                  {latin && <Text style={[T.caption, { color: C.brown, fontStyle: 'italic', opacity: 0.55, marginTop: 2 }]}>{latin}</Text>}
                                  <View style={{ flexDirection: 'row', alignItems: 'center', gap: 8, marginTop: 8 }}>
                                    <View style={[styles.confTrack, { flex: 1 }]}>
                                      <View style={[styles.confFill, { width: `${pct}%` }]} />
                                    </View>
                                    <Text style={[T.title, { color: C.green, minWidth: 42, textAlign: 'right' }]}>{pct}%</Text>
                                  </View>
                                </View>
                              </View>
                            );
                          })()}

                          {/* Expand/collapse other species */}
                          {prediction.all.length > 1 && (
                            <>
                              {expandedResults && (
                                <ScrollView style={{ maxHeight: 200, marginTop: 4 }} nestedScrollEnabled showsVerticalScrollIndicator={false}>
                                  {prediction.all.slice(1).map((item, idx) => {
                                    const pct = Math.round(item.confidence * 100);
                                    const { displayName: dn, latin: lt } = getNames(item.label);
                                    return (
                                      <View key={item.label}>
                                        {idx === 0 && (
                                          <View style={styles.leafDivider}>
                                            <View style={styles.leafLine} />
                                            <Text style={{ fontSize: 16 }}>🌿</Text>
                                            <View style={styles.leafLine} />
                                          </View>
                                        )}
                                        <View style={{ flexDirection: 'row', alignItems: 'center', gap: 14, marginBottom: 12 }}>
                                          <View style={styles.frogSquareSmall}>
                                            <Text style={{ fontSize: 28 }}>🐸</Text>
                                          </View>
                                          <View style={{ flex: 1 }}>
                                            <Text style={[T.title, { color: C.brown }]}>{dn}</Text>
                                            {lt && <Text style={[T.caption, { color: C.brown, fontStyle: 'italic', opacity: 0.55, marginTop: 2 }]}>{lt}</Text>}
                                            <View style={{ flexDirection: 'row', alignItems: 'center', gap: 8, marginTop: 6 }}>
                                              <View style={[styles.confTrack, { flex: 1 }]}>
                                                <View style={[styles.confFill, { width: `${pct}%`, backgroundColor: C.yellow }]} />
                                              </View>
                                              <Text style={[T.title, { color: C.yellowDark, minWidth: 42, textAlign: 'right' }]}>{pct}%</Text>
                                            </View>
                                          </View>
                                        </View>
                                      </View>
                                    );
                                  })}
                                </ScrollView>
                              )}
                              <TouchableOpacity onPress={() => setExpandedResults(e => !e)} activeOpacity={0.7} style={[styles.expandBtn, { justifyContent: 'center' }]}>
                                <Text style={[T.caption, { color: C.brownLight }]}>
                                  {expandedResults ? 'Hide others' : `${prediction.all.length - 1} others nearby`}
                                </Text>
                                <Ionicons name={expandedResults ? 'chevron-up' : 'chevron-down'} size={14} color={C.brownLight} />
                              </TouchableOpacity>
                            </>
                          )}
                        </View>
                      )}
                    </View>
                  </Animated.View>
                )}

                {status ? (
                  <Text style={[T.caption, { color: C.brownMid, textAlign: 'center', opacity: 0.7 }]}>{status}</Text>
                ) : null}
              </ScrollView>

              {/* Upload — always pinned at the bottom of the panel */}
              <View style={styles.uploadRow}>
                <View style={styles.dashedSeparator} />
                <TouchableOpacity
                  onPress={uploadAudio}
                  disabled={busy}
                  activeOpacity={0.7}
                  style={[styles.uploadBtn, busy && styles.uploadBtnDisabled]}
                >
                  <Text style={{ fontSize: 16 }}>☁️</Text>
                  <Text style={[T.medium, { color: C.brownMid, marginLeft: 8 }]}>Upload a recording</Text>
                </TouchableOpacity>
              </View>
            </View>
          </View>

          {/* ── PAGE 2: About ── */}
          <View style={styles.page}>
            <View style={{ paddingHorizontal: 24, paddingTop: 16, paddingBottom: 8 }}>
              <View style={[styles.fieldGuideBadge, { alignSelf: 'flex-start' }]}>
                <Text style={{ fontSize: 14 }}>🌿</Text>
                <Text style={[T.label, { color: C.brown, marginLeft: 6 }]}>LEARN MORE</Text>
              </View>
              <Text style={[T.h1, { color: C.brown, marginTop: 8 }]}>About</Text>
            </View>

            <ScrollView style={{ flex: 1 }} showsVerticalScrollIndicator={false} contentContainerStyle={{ padding: 24, gap: 16 }}>
              {/* Mission */}
              <View style={styles.aboutCard}>
                <Text style={[T.label, { color: C.green, marginBottom: 8 }]}>OUR MISSION</Text>
                <Text style={[T.body, { color: C.brownMid }]}>
                  FrogFinder uses on-device machine learning to identify frog species from their calls — no internet required.
                  By crowdsourcing detections, we help researchers track frog populations and detect early signs of habitat stress.
                </Text>
              </View>

              {/* Team */}
              <View style={styles.aboutCard}>
                <Text style={[T.label, { color: C.green, marginBottom: 12 }]}>THE TEAM</Text>
                {TEAM.map((member) => (
                  <View key={member.name} style={{ flexDirection: 'row', alignItems: 'center', marginBottom: 12 }}>
                    <View style={styles.avatar}>
                      <Text style={[T.title, { color: C.green }]}>{member.initials}</Text>
                    </View>
                    <View style={{ marginLeft: 12 }}>
                      <Text style={[T.title, { color: C.brown }]}>{member.name}</Text>
                      <Text style={[T.caption, { color: C.brownMid }]}>{member.role}</Text>
                    </View>
                  </View>
                ))}
              </View>

              {/* Sightings */}
              <View style={styles.aboutCard}>
                <Text style={[T.label, { color: C.green, marginBottom: 12 }]}>RECENT SIGHTINGS</Text>
                {SIGHTINGS.map((s, i) => (
                  <View key={i} style={[{ paddingBottom: 12 }, i < SIGHTINGS.length - 1 && styles.sightingDivider]}>
                    <View style={{ flexDirection: 'row', justifyContent: 'space-between', alignItems: 'center' }}>
                      <Text style={[T.medium, { color: C.brown, flex: 1, marginRight: 8 }]}>{s.species}</Text>
                      <View style={styles.badge}>
                        <Text style={[T.label, { color: C.green }]}>{s.confidence}%</Text>
                      </View>
                    </View>
                    <Text style={[T.caption, { color: C.brownMid, marginTop: 3 }]}>
                      {s.location} · {s.date}
                    </Text>
                  </View>
                ))}
              </View>
            </ScrollView>
          </View>
        </Animated.View>
        </View>

        {/* Tab bar — SafeAreaView fills the bottom home-indicator gap */}
        <SafeAreaView edges={['bottom']} style={{ backgroundColor: C.tabBg }}>
        <View style={styles.tabBar}>
          <Animated.View style={[styles.tabPill, {
            transform: [{ translateX: tabPillAnim.interpolate({ inputRange: [0, 1], outputRange: [4, SW / 2 + 4] }) }],
          }]} />
          {[
            { icon: 'leaf-outline',               label: 'Identify', page: 0 },
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
                <Ionicons name={icon} size={22} color={active ? C.green : C.brownLight} />
                <Text style={[T.caption, {
                  fontFamily: active ? 'Nunito-Bold' : 'Nunito-Regular',
                  fontSize: 11,
                  color: active ? C.green : C.brownLight,
                  marginTop: 3,
                }]}>
                  {label}
                </Text>
              </TouchableOpacity>
            );
          })}
        </View>
      </SafeAreaView>
    </SafeAreaView>
  );
}

export default function App() {
  return (
    <SafeAreaProvider>
      <AppContent />
    </SafeAreaProvider>
  );
}

const styles = StyleSheet.create({
  safeArea:    { flex: 1, backgroundColor: C.bg },
  pagesRow:    { position: 'absolute', top: 0, bottom: 0, left: 0, flexDirection: 'row', width: SW * 2 },
  page:        { width: SW, height: '100%' },
  hero:        { flex: 1, alignItems: 'center', justifyContent: 'center' },
  bottomPanel: { backgroundColor: C.surface, borderTopLeftRadius: 24, borderTopRightRadius: 24 },

  // Header badge + mascot
  fieldGuideBadge: {
    flexDirection: 'row', alignItems: 'center',
    backgroundColor: C.yellow, borderRadius: 99,
    paddingHorizontal: 14, paddingVertical: 7,
    shadowColor: C.brown, shadowOffset: { width: 0, height: 2 }, shadowRadius: 4, shadowOpacity: 0.15, elevation: 3,
  },
  mascotCircle: {
    width: 64, height: 64, borderRadius: 32,
    borderWidth: 3, borderColor: C.green,
    backgroundColor: C.greenLight,
    alignItems: 'center', justifyContent: 'center',
  },
  // "Ribbit!" pill overlapping the description box top-left
  ribbitLabel: {
    alignSelf: 'flex-start',
    backgroundColor: C.yellow,
    borderRadius: 99,
    paddingHorizontal: 13, paddingVertical: 5,
    marginLeft: 12, marginBottom: -12, zIndex: 1,
    shadowColor: C.brown, shadowOffset: { width: 0, height: 2 }, shadowRadius: 0, shadowOpacity: 0.2, elevation: 3,
  },
  descriptionBox: {
    backgroundColor: C.surface,
    borderWidth: 2, borderColor: C.brownLight,
    borderRadius: 16,
    padding: 12, paddingTop: 18,
    shadowColor: C.brownLight, shadowOffset: { width: 0, height: 3 }, shadowRadius: 0, shadowOpacity: 1, elevation: 4,
  },

  // Record button — square
  btnWrapper:  { width: 124, height: 128, alignItems: 'center', justifyContent: 'center' },
  ring: {
    position: 'absolute', width: 120, height: 120, borderRadius: 28,
    borderWidth: 2.5, borderColor: C.green,
  },
  recordBtn: {
    width: 120, height: 120, borderRadius: 28,
    alignItems: 'center', justifyContent: 'center',
    shadowOffset: { width: 0, height: 6 }, shadowRadius: 16, shadowOpacity: 0.25, elevation: 10,
  },
  recordBtnCancel:       { backgroundColor: C.tabBg,  shadowColor: C.brownLight },
  recordBtnIdle:         { backgroundColor: C.green,  shadowColor: C.green },
  recordBtnActive:       { backgroundColor: C.coral,  shadowColor: C.coral },
  recordBtnShadowIdle:   { backgroundColor: '#3d6b30' },
  recordBtnShadowActive: { backgroundColor: '#a05040' },
  recordBtnShadowCancel: { backgroundColor: C.brownLight },

  // Result card
  resultCard: {
    backgroundColor: C.surface,
    borderWidth: 1.5, borderColor: C.border,
    borderRadius: 22, overflow: 'hidden',
    shadowColor: C.brown, shadowOffset: { width: 0, height: 4 }, shadowRadius: 12, shadowOpacity: 0.1, elevation: 5,
  },
  resultCardHeader: {
    flexDirection: 'row', alignItems: 'center', justifyContent: 'space-between',
    backgroundColor: C.mint, paddingHorizontal: 16, paddingVertical: 12,
  },
  durationPill: {
    backgroundColor: C.white, borderRadius: 99,
    paddingHorizontal: 12, paddingVertical: 4,
    borderWidth: 1, borderColor: C.border,
  },
  frogSquareLarge: {
    width: 80, height: 80, borderRadius: 18,
    backgroundColor: C.greenLight,
    alignItems: 'center', justifyContent: 'center',
  },
  frogSquareSmall: {
    width: 62, height: 62, borderRadius: 14,
    backgroundColor: 'rgba(255,209,102,0.2)',
    borderWidth: 1.5, borderColor: 'rgba(196,168,130,0.3)',
    alignItems: 'center', justifyContent: 'center',
  },
  leafDivider:  { flexDirection: 'row', alignItems: 'center', gap: 10, marginVertical: 14 },
  leafLine:     { flex: 1, height: 1, backgroundColor: C.border },
  expandBtn:    { flexDirection: 'row', alignItems: 'center', gap: 6, paddingVertical: 8, justifyContent: 'center' },

  // Progress bars
  confTrack: { height: 8, backgroundColor: 'rgba(61,50,38,0.08)', borderRadius: 99, overflow: 'hidden' },
  confFill:  { height: '100%', backgroundColor: C.green, borderRadius: 99 },
  // (other species bars pass backgroundColor: C.yellow inline)

  // Upload
  uploadRow: { paddingHorizontal: 20, paddingBottom: 20, paddingTop: 4 },
  dashedSeparator: {
    borderWidth: 1.5, borderColor: C.border, borderStyle: 'dashed',
    borderRadius: 1, marginBottom: 12,
  },
  uploadBtn: {
    flexDirection: 'row', alignItems: 'center', justifyContent: 'center',
    paddingVertical: 13, borderRadius: 16,
    backgroundColor: C.surface,
    borderWidth: 1.5, borderColor: C.border,
  },
  uploadBtnDisabled: { opacity: 0.4 },

  // Tab bar
  tabBar:  { flexDirection: 'row', borderTopWidth: 1, borderTopColor: C.border, backgroundColor: C.tabBg },
  tabItem: { flex: 1, alignItems: 'center', justifyContent: 'center', paddingVertical: 10, zIndex: 1 },
  tabPill: { position: 'absolute', top: 6, bottom: 6, width: SW / 2 - 8, borderRadius: 12, backgroundColor: C.greenLight },

  // About page
  aboutCard: {
    backgroundColor: C.surface, borderWidth: 1, borderColor: C.border, borderRadius: 20, padding: 18,
    shadowColor: C.brown, shadowOffset: { width: 0, height: 3 }, shadowRadius: 8, shadowOpacity: 0.06, elevation: 3,
  },
  avatar:          { width: 42, height: 42, borderRadius: 12, backgroundColor: C.greenLight, alignItems: 'center', justifyContent: 'center' },
  badge:           { backgroundColor: C.greenLight, borderRadius: 10, paddingHorizontal: 8, paddingVertical: 3 },
  sightingDivider: { borderBottomWidth: 1, borderBottomColor: C.border, marginBottom: 12 },

  // AC dialogue box
  acDialogue: {
    backgroundColor: C.surface,
    borderWidth: 2.5, borderColor: C.brownLight,
    borderRadius: 18, padding: 16,
    shadowColor: C.brownLight,
    shadowOffset: { width: 0, height: 4 }, shadowRadius: 0, shadowOpacity: 1,
    elevation: 6,
  },

  // Clouds
  cloud: { position: 'absolute' },
});

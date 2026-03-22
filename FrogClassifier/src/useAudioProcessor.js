import { useState, useRef } from 'react';
import { Animated, Easing, LayoutAnimation } from 'react-native';
import { useAudioRecorder, AudioModule, useAudioPlayer, useAudioPlayerStatus } from 'expo-audio';
import * as Haptics from 'expo-haptics';
import * as FileSystem from 'expo-file-system/legacy';
import * as DocumentPicker from 'expo-document-picker';
import { Buffer } from 'buffer';

import { SR, WIN_SAMPLES, STRIDE_SAMPLES, RECORDING_OPTIONS } from './constants';

// Windows whose RMS energy falls below this are classified as Background
// without running the SVM — saves full feature extraction on silent segments.
// ~0.005 ≈ -46 dBFS; well below any frog call but above digital silence.
const RMS_SILENCE_THRESHOLD = 0.005;
import { predict, majorityVote, MODEL_NAMES } from './classifier';
import { extractFeatures } from './features';
import { parseWav, decodePcm, resample } from './audioUtils';

const LAYOUT_MOVE = { duration: 320, update: { type: 'easeInEaseOut' } };

export function useAudioProcessor(selectedModel = MODEL_NAMES[0]) {
  const audioRecorder = useAudioRecorder(RECORDING_OPTIONS);
  const player        = useAudioPlayer(null);
  const playerStatus  = useAudioPlayerStatus(player);

  const [prediction,      setPrediction]      = useState(null);
  const [status,          setStatus]          = useState('');
  const [progress,        setProgress]        = useState(null);
  const [audioDuration,   setAudioDuration]   = useState(0);
  const [isProcessing,    setIsProcessing]    = useState(false);
  const [expandedResults, setExpandedResults] = useState(false);
  const [audioUri,        setAudioUri]        = useState(null);

  const cancelRef        = useRef(false);
  const progressAnim     = useRef(new Animated.Value(0)).current;
  const progressFadeAnim = useRef(new Animated.Value(0)).current;
  const resultDropAnim   = useRef(new Animated.Value(1)).current;

  // -------------------------------------------------------------------------
  // dropAndClearPrediction
  // -------------------------------------------------------------------------
  const dropAndClearPrediction = (cb) => {
    if (!prediction) { cb?.(); return; }
    Animated.timing(resultDropAnim, { toValue: 0, duration: 200, useNativeDriver: true }).start(() => {
      LayoutAnimation.configureNext(LAYOUT_MOVE);
      setPrediction(null);
      cb?.();
    });
  };

  // -------------------------------------------------------------------------
  // playAudio — toggle playback of the last recorded / uploaded file
  // -------------------------------------------------------------------------
  const playAudio = async () => {
    if (!audioUri) return;
    try {
      if (playerStatus.playing) {
        player.pause();
      } else {
        // Switch to media playback mode so audio routes to the main speaker,
        // not the earpiece. Recording sets allowsRecording:true which puts iOS
        // into voice processing mode and locks routing to the earpiece.
        await AudioModule.setAudioModeAsync({ allowsRecording: false, playsInSilentMode: true });
        player.replace({ uri: audioUri });
        player.play();
      }
    } catch (err) {
      setStatus(`Playback error: ${err.message}`);
    }
  };

  // -------------------------------------------------------------------------
  // processAudio — WAV decode → feature extraction → SVM classify
  // -------------------------------------------------------------------------
  const processAudio = async (uri) => {
    // Stop any active playback before processing
    if (playerStatus.playing) player.pause();

    setAudioUri(uri);
    setIsProcessing(true);
    setStatus('Reading file...');

    const base64 = await FileSystem.readAsStringAsync(uri, { encoding: 'base64' });
    const bytes  = Buffer.from(base64, 'base64');

    setStatus('Decoding PCM...');
    const { dataOffset, fileSr, bitsPerSample, numChannels } = parseWav(bytes);

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

    const mono      = decodePcm(bytes, dataOffset, numChannels);
    const audioData = resample(mono, fileSr, SR);

    setStatus('Classifying...');
    const totalWindows = Math.max(0, Math.floor((audioData.length - WIN_SAMPLES) / STRIDE_SAMPLES) + 1);

    progressFadeAnim.setValue(0);
    LayoutAnimation.configureNext(LAYOUT_MOVE);
    setProgress(0);
    progressAnim.setValue(0);
    cancelRef.current = false;
    Animated.timing(progressFadeAnim, { toValue: 1, duration: 220, useNativeDriver: true }).start();

    const benchStart  = Date.now();
    predict(extractFeatures(audioData.slice(0, WIN_SAMPLES)), selectedModel);
    const estimatedMs = Math.max(1500, (Date.now() - benchStart) * totalWindows * 1.3);

    const progAnim = Animated.timing(progressAnim, {
      toValue:  0.92,
      duration: estimatedMs,
      easing:   Easing.out(Easing.cubic),
      useNativeDriver: true,
    });
    progAnim.start();

    const windowPredictions = [];
    const labelCounts = {}; // non-Background label → vote count for early stopping
    let windowIdx = 0;

    for (let start = 0; start + WIN_SAMPLES <= audioData.length; start += STRIDE_SAMPLES) {
      if (cancelRef.current) break;

      const window = audioData.slice(start, start + WIN_SAMPLES);

      // RMS pre-filter — skip SVM entirely for silent windows
      let sumSq = 0;
      for (let k = 0; k < window.length; k++) sumSq += window[k] * window[k];
      const rms = Math.sqrt(sumSq / window.length);

      let pred;
      if (rms < RMS_SILENCE_THRESHOLD) {
        pred = { label: 'Background', confidence: 1 };
      } else {
        pred = predict(extractFeatures(window), selectedModel);
      }

      windowPredictions.push(pred);
      windowIdx++;
      setProgress(windowIdx / totalWindows);

      // Update incremental vote counts for early stopping
      if (pred.label !== 'Background') {
        labelCounts[pred.label] = (labelCounts[pred.label] || 0) + 1;
      }

      // Early stopping — if the leading label can't be beaten by all remaining
      // windows going to 2nd place, the result is already decided
      const remaining = totalWindows - windowIdx;
      const sorted = Object.values(labelCounts).sort((a, b) => b - a);
      if (sorted.length >= 1 && sorted[0] - (sorted[1] ?? 0) > remaining) break;

      if (windowIdx % 5 === 0) await new Promise(r => setTimeout(r, 0));
    }

    progAnim.stop();
    await new Promise(r =>
      Animated.timing(progressAnim, { toValue: 1, duration: 300, easing: Easing.out(Easing.quad), useNativeDriver: true }).start(r)
    );
    await new Promise(r =>
      Animated.timing(progressFadeAnim, { toValue: 0, duration: 240, useNativeDriver: true }).start(r)
    );
    LayoutAnimation.configureNext(LAYOUT_MOVE);
    setProgress(null);

    setIsProcessing(false);
    if (cancelRef.current) { setStatus('Cancelled'); return; }
    if (!windowPredictions.length) { setStatus('Audio too short to classify'); return; }

    setAudioDuration(Math.round(audioData.length / SR));
    const result = majorityVote(windowPredictions);

    setExpandedResults(false);
    resultDropAnim.setValue(0);
    LayoutAnimation.configureNext(LAYOUT_MOVE);
    setPrediction(result);
    Animated.timing(resultDropAnim, { toValue: 1, duration: 320, useNativeDriver: true }).start();
    setStatus('');

    Haptics.notificationAsync(
      result.label !== 'Background'
        ? Haptics.NotificationFeedbackType.Success
        : Haptics.NotificationFeedbackType.Warning
    );
  };

  // -------------------------------------------------------------------------
  // startRecording / stopRecording / uploadAudio
  // -------------------------------------------------------------------------
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

  const uploadAudio = async () => {
    try {
      const result = await DocumentPicker.getDocumentAsync({ type: 'audio/wav', copyToCacheDirectory: true });
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

  return {
    prediction, status, progress, audioDuration, isProcessing,
    expandedResults, setExpandedResults,
    isRecording: audioRecorder.isRecording,
    audioUri,
    isPlaying: playerStatus.playing,
    progressAnim, progressFadeAnim, resultDropAnim,
    startRecording, stopRecording, uploadAudio, playAudio,
    cancel: () => { cancelRef.current = true; },
  };
}

import { useState } from 'react';
import { View, Text, Button, StyleSheet } from 'react-native';
import { useAudioRecorder, AudioModule, RecordingPresets } from 'expo-audio';
import * as FileSystem from 'expo-file-system';
import { Buffer } from 'buffer';
import Meyda from 'meyda';

const model = require('./assets/svm_model.json');

function rbfKernel(x, y, gamma) {
  const diff = x.map((val, i) => val - y[i]);
  const norm = diff.reduce((sum, val) => sum + val * val, 0);
  return Math.exp(-gamma * norm);
}

function predict(features) {
  const scaled = features.map((val, i) => (val - model.scaler_mean[i]) / model.scaler_scale[i]);

  const decisions = model.intercept.map((inter, k) => {
    let sum = inter;
    for (let i = 0; i < model.support_vectors.length; i++) {
      sum += model.dual_coef[k][i] * rbfKernel(scaled, model.support_vectors[i], model.gamma);
    }
    return sum;
  });

  const maxIndex = decisions.indexOf(Math.max(...decisions));
  return model.classes[maxIndex];
}

export default function App() {
  const audioRecorder = useAudioRecorder(RecordingPresets.HIGH_QUALITY);
  const [prediction, setPrediction] = useState('');
  const [status, setStatus] = useState('');

  const startRecording = async () => {
    try {
      setStatus('Requesting permission...');
      const { granted } = await AudioModule.requestRecordingPermissionsAsync();
      if (!granted) {
        setStatus('Microphone permission denied');
        return;
      }
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
      await audioRecorder.stop();
      await processAudio(audioRecorder.uri);
    } catch (err) {
      setStatus(`Error: ${err.message}`);
      console.error('Failed to stop recording', err);
    }
  };

  const processAudio = async (uri) => {
    const file = await FileSystem.readAsStringAsync(uri, { encoding: FileSystem.EncodingType.Base64 });
    const buffer = Buffer.from(file, 'base64');
    const audioData = new Float32Array(buffer.length / 2);
    for (let i = 0; i < audioData.length; i++) {
      audioData[i] = buffer.readInt16LE(i * 2) / 32768;
    }

    const frameSize = 512;
    const hopSize = 256;
    const mfccs = [];
    for (let i = 0; i < audioData.length - frameSize; i += hopSize) {
      const frame = audioData.slice(i, i + frameSize);
      const mfcc = Meyda.extract('mfcc', frame, { sampleRate: 44100 });
      mfccs.push(mfcc);
    }

    const flatFeatures = mfccs.slice(0, 47).flat();
    while (flatFeatures.length < 616) flatFeatures.push(0);
    flatFeatures.splice(616);

    const pred = predict(flatFeatures);
    setPrediction(pred);
    setStatus('Done');
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
  container: {
    flex: 1,
    backgroundColor: '#fff',
    alignItems: 'center',
    justifyContent: 'center',
    gap: 16,
  },
  title: {
    fontSize: 24,
    fontWeight: 'bold',
  },
  status: {
    color: '#666',
  },
  prediction: {
    fontSize: 18,
    fontWeight: '600',
  },
});

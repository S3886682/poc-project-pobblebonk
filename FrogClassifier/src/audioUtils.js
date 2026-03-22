import { SR } from './constants';

// ---------------------------------------------------------------------------
// WAV header parser — finds the PCM data chunk offset plus format metadata.
// Never assumes a fixed 44-byte header.
// ---------------------------------------------------------------------------
export function parseWav(bytes) {
  let dataOffset = -1, fileSr = SR, bitsPerSample = 16, numChannels = 1;
  try {
    let pos = 12; // skip RIFF(4) + fileSize(4) + WAVE(4)
    while (pos + 8 <= bytes.length) {
      const chunkId   = bytes.toString('ascii', pos, pos + 4);
      const chunkSize = bytes.readUInt32LE(pos + 4);
      if (chunkId === 'fmt ') {
        numChannels   = bytes.readUInt16LE(pos + 10);
        fileSr        = bytes.readUInt32LE(pos + 12);
        bitsPerSample = bytes.readUInt16LE(pos + 22);
      } else if (chunkId === 'data') {
        dataOffset = pos + 8;
        break;
      }
      pos += 8 + chunkSize + (chunkSize % 2); // chunks are word-aligned
    }
  } catch (_) {}
  return { dataOffset, fileSr, bitsPerSample, numChannels };
}

// ---------------------------------------------------------------------------
// Decode first channel of 16-bit PCM to normalised Float32Array [-1, 1]
// ---------------------------------------------------------------------------
export function decodePcm(bytes, dataOffset, numChannels) {
  const bytesPerSample = 2; // 16-bit
  const nSamples = Math.floor((bytes.length - dataOffset) / (bytesPerSample * numChannels));
  const out = new Float32Array(nSamples);
  for (let i = 0; i < nSamples; i++) {
    out[i] = bytes.readInt16LE(dataOffset + i * bytesPerSample * numChannels) / 32768;
  }
  return out;
}

// ---------------------------------------------------------------------------
// Linear interpolation resampler — only runs when file SR !== target SR
// ---------------------------------------------------------------------------
export function resample(mono, fileSr, targetSr) {
  if (fileSr === targetSr) return mono;
  const ratio  = fileSr / targetSr;
  const newLen = Math.round(mono.length / ratio);
  const out    = new Float32Array(newLen);
  for (let i = 0; i < newLen; i++) {
    const pos  = i * ratio;
    const idx  = Math.floor(pos);
    const frac = pos - idx;
    const a    = mono[Math.min(idx,     mono.length - 1)];
    const b    = mono[Math.min(idx + 1, mono.length - 1)];
    out[i] = a + frac * (b - a);
  }
  return out;
}

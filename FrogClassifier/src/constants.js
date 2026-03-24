export const SR            = 32000;
export const WIN_SAMPLES   = Math.round(SR * 0.3);    // 9600
export const STRIDE_SAMPLES = Math.round(SR * 0.2);   // 6400
export const N_MFCC        = 40;
export const FFT_SIZE      = 2048;
export const HOP_SIZE      = 512;

export const RECORDING_OPTIONS = {
  extension: '.wav',
  sampleRate: SR,
  numberOfChannels: 1,
  bitRate: SR * 16,
  isMeteringEnabled: true,
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

export const TEAM = [
  { name: 'Ashley', role: 'Project Manager',       initials: 'A' },
  { name: 'Bridget', role: 'UI Designer',   initials: 'B' },
  { name: 'Daniel',    role: 'ML Engineer',     initials: 'D' },
];

export const SIGHTINGS = [
  { species: 'Litoria caerulea',      location: 'Brisbane, QLD',   date: '18 Mar 2026', confidence: 94 },
  { species: 'Litoria fallax',        location: 'Gold Coast, QLD', date: '15 Mar 2026', confidence: 87 },
  { species: 'Crinia signifera',      location: 'Sydney, NSW',     date: '12 Mar 2026', confidence: 91 },
  { species: 'Uperoleia laevigata',   location: 'Darwin, NT',      date: '10 Mar 2026', confidence: 78 },
  { species: 'Limnodynastes peronii', location: 'Melbourne, VIC',  date: '8 Mar 2026',  confidence: 85 },
];

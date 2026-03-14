"""
Retrain SVM with 40-MFCC features to match Meyda's output.

Meyda caps MFCC at 40 coefficients regardless of configuration.
This script extracts the same 256-feature vector the JS app produces:
  - MFCC (40 coeff) × mean+std  = 80
  - Delta-1         × mean+std  = 80
  - Delta-2         × mean+std  = 80
  - Spectral contrast (7 bands) × mean+std = 14
  - Spectral centroid × mean+std = 2
  Total = 256

Outputs: svm_model_meyda.json (drop into FrogClassifier/assets/svm_model.json)
"""

import os
import json
import numpy as np
import librosa
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import joblib

# ── Parameters (must match App.js constants) ──────────────────────────────────
SR           = 32000
WIN_SEC      = 0.3
STRIDE_SEC   = 0.2
N_MFCC       = 40      # Meyda's actual output cap
N_FFT        = 2048
HOP_LENGTH   = 512

BASE_DIR      = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'project-pobblebonk', 'backend'))
TRAINING_DIR  = os.path.join(BASE_DIR, 'Training Audio')
BACKGROUND_DIR = os.path.join(BASE_DIR, 'Background Audio')
OUTPUT_JSON   = os.path.join(os.path.dirname(__file__), 'svm_model.json')
OUTPUT_PKL    = os.path.join(os.path.dirname(__file__), 'Trained Models', 'svm_classifier_meyda.pkl')

WIN_SAMPLES    = int(SR * WIN_SEC)
STRIDE_SAMPLES = int(SR * STRIDE_SEC)

# ── Feature extraction ─────────────────────────────────────────────────────────

def extract_features(y, sr=SR, n_mfcc=N_MFCC, n_fft=N_FFT, hop_length=HOP_LENGTH):
    mfcc     = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc, n_fft=n_fft, hop_length=hop_length)
    delta1   = librosa.feature.delta(mfcc, order=1)
    delta2   = librosa.feature.delta(mfcc, order=2)
    contrast = librosa.feature.spectral_contrast(y=y, sr=sr, n_fft=n_fft, hop_length=hop_length)
    centroid = librosa.feature.spectral_centroid(y=y, sr=sr, n_fft=n_fft, hop_length=hop_length)

    feats = []
    for m in [mfcc, delta1, delta2, contrast, centroid]:
        feats.append(np.mean(m, axis=1))
        feats.append(np.std(m, axis=1))
    return np.hstack(feats)

def extract_windows(file_path, label, features, labels):
    try:
        y, _ = librosa.load(file_path, sr=SR)
    except Exception as e:
        print(f"  [SKIP] {file_path}: {e}")
        return

    for start in range(0, len(y) - WIN_SAMPLES + 1, STRIDE_SAMPLES):
        seg = y[start:start + WIN_SAMPLES]
        feat = extract_features(seg)
        features.append(feat)
        labels.append(label)

# ── Load data ─────────────────────────────────────────────────────────────────

features, labels = [], []

print("Extracting training audio windows...")
for species in sorted(os.listdir(TRAINING_DIR)):
    species_dir = os.path.join(TRAINING_DIR, species)
    if not os.path.isdir(species_dir):
        continue
    print(f"  {species}")
    for fname in os.listdir(species_dir):
        if fname.lower().endswith('.wav'):
            extract_windows(os.path.join(species_dir, fname), species, features, labels)

print("Extracting background audio windows...")
if os.path.isdir(BACKGROUND_DIR):
    for fname in os.listdir(BACKGROUND_DIR):
        if fname.lower().endswith('.wav'):
            extract_windows(os.path.join(BACKGROUND_DIR, fname), 'Background', features, labels)

X = np.array(features)
y = np.array(labels)
print(f"\nTotal samples: {X.shape[0]}, features: {X.shape[1]}")

# ── Train ─────────────────────────────────────────────────────────────────────

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

clf = Pipeline([
    ('scaler', StandardScaler()),
    ('svc',    SVC(kernel='rbf', C=10, gamma='scale', random_state=42)),
])

print("Training SVM...")
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# ── Save pkl ──────────────────────────────────────────────────────────────────

os.makedirs(os.path.dirname(OUTPUT_PKL), exist_ok=True)
joblib.dump(clf, OUTPUT_PKL)
print(f"Model saved to {OUTPUT_PKL}")

# ── Export JSON for the JS app ────────────────────────────────────────────────

scaler = clf.named_steps['scaler']
svc    = clf.named_steps['svc']

data = {
    'kernel':       svc.kernel,
    'gamma':        float(svc._gamma),
    'support_vectors': svc.support_vectors_.tolist(),
    'dual_coef':    svc.dual_coef_.tolist(),
    'intercept':    svc.intercept_.tolist(),
    'classes':      svc.classes_.tolist(),
    'n_support':    svc.n_support_.tolist(),
    'scaler_mean':  scaler.mean_.tolist(),
    'scaler_scale': scaler.scale_.tolist(),
}

with open(OUTPUT_JSON, 'w') as f:
    json.dump(data, f)

print(f"JSON exported to {OUTPUT_JSON}")
print(f"\nFeature vector size: {X.shape[1]}")
print(f"Classes ({len(svc.classes_)}): {list(svc.classes_)}")

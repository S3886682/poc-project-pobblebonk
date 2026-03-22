"""
Train SVM from features pre-extracted by Meyda (extract_features_meyda/extract.js).

Run AFTER: cd extract_features_meyda && npm install && node extract.js

This guarantees that training features exactly match what the JS app produces
at inference time.

Changes vs original:
  - SVM_C reduced from 10 -> 1 (fewer support vectors -> faster on-device inference).
    Compare the classification report to the old C=10 run before deploying.
    If per-class F1 drops >2–3 points on important species, try C=5 as a midpoint.
  - probability=True adds Platt sigmoid calibration via 5-fold CV.
    Training is ~5x slower but the JSON output gains prob_a/prob_b arrays that
    the JS app uses for Wu et al. pairwise coupling -> proper 0–1 probabilities.
  - per_class_f1 is exported so the app can display model accuracy per species.
"""

import json
import numpy as np
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import joblib
import os

# ── Tunable parameters ────────────────────────────────────────────────────────
# Lower C -> fewer SVs -> faster inference, but slightly lower accuracy.
# Previous C=10 model had 4602 SVs. C=1 typically produces 40–60% fewer.
SVM_C = 10

INPUT_JSON  = os.path.join(os.path.dirname(__file__), 'features_meyda.json')
OUTPUT_JSON = os.path.join(os.path.dirname(__file__), 'svm_model.json')
OUTPUT_PKL  = os.path.join(os.path.dirname(__file__), 'Trained Models', 'svm_classifier_meyda.pkl')

print(f'Loading features from {INPUT_JSON} ...')
with open(INPUT_JSON) as f:
    data = json.load(f)

X = np.array(data['features'], dtype=np.float32)
y = np.array(data['labels'])
print(f'Loaded {X.shape[0]} samples, {X.shape[1]} features, {len(set(y))} classes')

# Drop rows with NaN or Inf (can arise from silent segments)
valid = np.isfinite(X).all(axis=1)
dropped = (~valid).sum()
if dropped:
    print(f'Dropping {dropped} invalid samples (NaN/Inf)')
X, y = X[valid], y[valid]
print(f'Remaining: {X.shape[0]} samples, {len(set(y))} classes')

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

clf = Pipeline([
    ('scaler', StandardScaler()),
    ('svc',    SVC(kernel='rbf', C=SVM_C, gamma='scale', random_state=42, probability=True)),
])

print(f'Training SVM (C={SVM_C}, probability=True) ...')
print('Note: probability=True runs 5-fold CV internally — this takes longer than usual.')
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)
print('\nClassification Report:')
print(classification_report(y_test, y_pred))

scaler = clf.named_steps['scaler']
svc    = clf.named_steps['svc']

total_svs = int(svc.n_support_.sum())
print(f'Support vectors: {total_svs}  (previous C=10 model: 4602)')
print(f'Inference speedup estimate: {4602 / total_svs:.1f}x from SV reduction alone')

# ── Save pkl ──────────────────────────────────────────────────────────────────
os.makedirs(os.path.dirname(OUTPUT_PKL), exist_ok=True)
joblib.dump(clf, OUTPUT_PKL)
print(f'Model saved -> {OUTPUT_PKL}')

# ── Per-class F1 from classification report ───────────────────────────────────
report = classification_report(y_test, y_pred, output_dict=True)
per_class_f1 = {}
for cls in svc.classes_:
    if cls in report:
        per_class_f1[cls] = round(report[cls]['f1-score'], 3)

# ── Export JSON for the JS app ────────────────────────────────────────────────
export = {
    'kernel':          svc.kernel,
    'gamma':           float(svc._gamma),
    'support_vectors': svc.support_vectors_.tolist(),
    'dual_coef':       svc.dual_coef_.tolist(),
    'intercept':       svc.intercept_.tolist(),
    'classes':         svc.classes_.tolist(),
    'n_support':       svc.n_support_.tolist(),
    'scaler_mean':     scaler.mean_.tolist(),
    'scaler_scale':    scaler.scale_.tolist(),
    # Platt calibration: one (A, B) pair per pairwise classifier,
    # ordered identically to intercept (i.e. pair (0,1),(0,2),...,(k-2,k-1)).
    'prob_a':          svc.probA_.tolist(),
    'prob_b':          svc.probB_.tolist(),
    # Per-species F1 score on the held-out test set (0–1).
    'per_class_f1':    per_class_f1,
}

with open(OUTPUT_JSON, 'w') as f:
    json.dump(export, f)
print(f'JSON exported -> {OUTPUT_JSON}')
print(f'\nFeature vector size: {X.shape[1]}')
print(f'Classes ({len(svc.classes_)}): {list(svc.classes_)}')
print(f'\nNow copy {OUTPUT_JSON} -> FrogClassifier/assets/svm_model.json')

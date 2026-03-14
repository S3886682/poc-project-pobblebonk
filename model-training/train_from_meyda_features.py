"""
Train SVM from features pre-extracted by Meyda (extract_features_meyda/extract.js).

Run AFTER: cd extract_features_meyda && npm install && node extract.js

This guarantees that training features exactly match what the JS app produces
at inference time.
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
    ('svc',    SVC(kernel='rbf', C=10, gamma='scale', random_state=42)),
])

print('Training SVM...')
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)
print('\nClassification Report:')
print(classification_report(y_test, y_pred))

os.makedirs(os.path.dirname(OUTPUT_PKL), exist_ok=True)
joblib.dump(clf, OUTPUT_PKL)
print(f'Model saved → {OUTPUT_PKL}')

# Export JSON for App.js
scaler = clf.named_steps['scaler']
svc    = clf.named_steps['svc']

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
}

with open(OUTPUT_JSON, 'w') as f:
    json.dump(export, f)
print(f'JSON exported → {OUTPUT_JSON}')
print(f'\nFeature vector size: {X.shape[1]}')
print(f'Classes ({len(svc.classes_)}): {list(svc.classes_)}')
print(f'\nNow copy {OUTPUT_JSON} → FrogClassifier/assets/svm_model.json')

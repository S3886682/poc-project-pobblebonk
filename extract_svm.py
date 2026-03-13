import joblib
import numpy as np
import json

# Load the model
model = joblib.load('Trained Models/svm_classifier.pkl')

scaler = model.named_steps['scaler']
svc = model.named_steps['svc']

# Prepare data for JS
data = {
    'kernel': svc.kernel,
    'gamma': svc.gamma,
    'support_vectors': svc.support_vectors_.tolist(),
    'dual_coef': svc.dual_coef_.tolist(),
    'intercept': svc.intercept_.tolist(),
    'classes': svc.classes_.tolist(),
    'scaler_mean': scaler.mean_.tolist(),
    'scaler_scale': scaler.scale_.tolist()
}

with open('svm_model.json', 'w') as f:
    json.dump(data, f)

print("Model exported to svm_model.json")
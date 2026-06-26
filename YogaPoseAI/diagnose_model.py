import pandas as pd
import numpy as np
import torch
from backend.model_loader import model, scaler, label_map

# Load dataset
df = pd.read_csv('final_dataset.csv')
X = df.drop(columns=['label']).values
y = df['label'].astype(int).values

# Sample a few records per class
print("=" * 70)
print("MODEL PREDICTION DIAGNOSTIC")
print("=" * 70)
print(f"Dataset shape: {X.shape}")
print(f"Scaler mean: {scaler.mean_[:5]}... (first 5)")
print(f"Scaler scale: {scaler.scale_[:5]}... (first 5)")
print()

X_scaled = scaler.transform(X)

# Get all predictions
with torch.no_grad():
    inputs = torch.FloatTensor(X_scaled)
    outputs = model(inputs)
    probs = torch.softmax(outputs, dim=1).cpu().numpy()
    preds = outputs.argmax(dim=1).cpu().numpy()

print("PREDICTION DISTRIBUTION:")
unique, counts = np.unique(preds, return_counts=True)
for u, c in zip(unique, counts):
    pct = 100 * c / len(preds)
    print(f"  Class {u} ({label_map.get(int(u), '?')}): {c:5d} samples ({pct:5.1f}%)")

print("\nOUTPUT PROBABILITY RANGE:")
print(f"  Min prob: {probs.min():.6f}")
print(f"  Max prob: {probs.max():.6f}")
print(f"  Mean prob per class: {probs.mean(axis=0)}")

print("\nSAMPLE PREDICTIONS (first 20):")
print(f"{'True Class':<20} {'Pred Class':<20} {'Max Prob':<10} {'Top 3 Probs':<50}")
for i in range(min(20, len(y))):
    true_cls = label_map.get(int(y[i]), '?')
    pred_cls = label_map.get(int(preds[i]), '?')
    top3_idx = np.argsort(probs[i])[-3:][::-1]
    top3_str = ', '.join([f"{label_map.get(j, '?')}:{probs[i,j]:.3f}" for j in top3_idx])
    print(f"{true_cls:<20} {pred_cls:<20} {probs[i, preds[i]]:<10.4f} {top3_str:<50}")

print("\nMODEL WEIGHTS RANGE:")
for name, param in model.named_parameters():
    print(f"  {name}: min={param.min():.4f}, max={param.max():.4f}, mean={param.mean():.4f}")

import pandas as pd
import numpy as np
import torch
from backend.model_loader import model, scaler, label_map
from sklearn.metrics import classification_report, confusion_matrix

print('Loading dataset...')
df = pd.read_csv('final_dataset.csv')
X = df.drop(columns=['label']).values
y = df['label'].astype(int).values

print('Scaling...')
X_scaled = scaler.transform(X)

print('Running model inference...')
with torch.no_grad():
    inputs = torch.FloatTensor(X_scaled)
    outputs = model(inputs)
    preds = torch.argmax(outputs, dim=1).cpu().numpy()

print('Dataset size:', X.shape)
print('\nLabel mapping (index: name):')
for k,v in label_map.items():
    print(k, v)

print('\nClassification report:')
print(classification_report(y, preds, zero_division=0))

cm = confusion_matrix(y, preds)
print('\nConfusion matrix (rows=true, cols=pred):')
print(cm)

unique, counts = np.unique(preds, return_counts=True)
print('\nPrediction distribution:')
for u,c in zip(unique, counts):
    print(u, label_map.get(int(u),'?'), c)

from collections import Counter
cnt = Counter(preds)
most = cnt.most_common(1)[0]
print('\nMost predicted class:', most, label_map.get(int(most[0]),'?'))

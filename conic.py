from kan import *
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT_DIR = Path(__file__).resolve().parent
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

df = pd.read_csv(ROOT_DIR / 'dataset' / 'Conic-Section_dataset.csv')

features = df.drop(columns=['shape']).to_numpy(dtype=np.float32)
labels, shape_names = pd.factorize(df['shape'])
labels = labels.astype(np.int64)

rng = np.random.default_rng(seed=1)
indices = rng.permutation(len(df))
test_size = int(0.2 * len(df))
test_indices = indices[:test_size]
train_indices = indices[test_size:]

train_input = torch.tensor(features[train_indices], device=device)
train_label = torch.tensor(labels[train_indices], device=device)
test_input = torch.tensor(features[test_indices], device=device)
test_label = torch.tensor(labels[test_indices], device=device)

dataset = {
    'train_input': train_input,
    'train_label': train_label,
    'test_input': test_input,
    'test_label': test_label,
}

model = KAN(width=[12, 6, len(shape_names)], grid=3, k=3, seed=1, device=device)

results = model.fit(
    dataset,
    opt='LBFGS',
    steps=100,
    loss_fn=torch.nn.CrossEntropyLoss(),
)

print('Classi:', dict(enumerate(shape_names)))
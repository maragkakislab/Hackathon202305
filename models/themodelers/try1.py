#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 23 16:18:23 2023

@author: maragkakise2
"""


#import os
import pandas as pd
import torch
from scipy.stats import zscore
from torch import nn, optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
import torch.nn as nn
import torch.nn.functional as F
from torchmetrics import Accuracy

df = pd.read_csv("prep/experiments/mmu_dRNA_3T3_PION_1/joined_features_expanded_aggmax.tab")

# Drop unused columns
drop_cols = ['transcript', "name", 'gene', 'transcript_id']
df.drop(drop_cols, axis=1, inplace=True)

# Drop row with no target value
df = df[df['t5'] != '--']
df['t5'] = pd.to_numeric(df['t5'])
df = df[(df['t5'] >= 0.1) & (df['t5'] <= 30)]


# Normalize count columns by the column sum
norm_cols = ['n_readlength', 'n_polyA']
for c in norm_cols:
    df[c] /= df[c].sum()

# Extract our target column from the dataframe
target = df[['t5']]
df.drop(['t5'], axis=1, inplace=True)


# Split data into training and testing set
X_train, X_test, y_train, y_test = train_test_split(df, target,
                                                    test_size=0.2,
                                                    random_state=42)


# Create batches
class MyDataset(Dataset):
    def __init__(self, df_x, df_y, normalize=True):
        self.x = df_x.values
        self.y = df_y.values

        # Standardize the input features
        if normalize:
            scaler = StandardScaler()
            self.x = scaler.fit_transform(self.x)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        x = torch.tensor(self.x[idx], dtype=torch.float32)
        y = torch.tensor(self.y[idx], dtype=torch.float32)

        return x, y


class SimpleNet(pl.LightningModule):
    def __init__(self, input_dim, hidden_dim, hidden_dim2, hidden_dim3, output_dim):
        super(SimpleNet, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim2),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(hidden_dim2, hidden_dim3),
            nn.ReLU(),
            nn.Linear(hidden_dim3, output_dim)
            )
        # self.accuracy = Accuracy(task='multiclass', num_classes=num_classes)

    def forward(self, x):
        return self.net(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        pred = self(x)
        loss = F.mse_loss(pred, y)
        self.log('train_loss', loss, prog_bar=True)

        # accuracy = self.accuracy(pred, y)
        # self.log('train_accuracy', accuracy, prog_bar=True)

        return loss

    def predict_step(self, batch, batch_idx):
        x, y = batch
        pred = self(x)
        return pred, y
        # metrics = {'accuracy':self.accuracy(pred, y)}
        # self.log_dict(metrics)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.001, weight_decay=0.1)


# Define the dimensions of the network
input_dim = 30
hidden_dim = 50
hidden_dim2 = 30
hidden_dim3 = 10
output_dim = 1
batch_size = 8

dset_train = MyDataset(X_train, y_train, normalize=True)
dset_test = MyDataset(X_test, y_test, normalize=True)

train_loader = DataLoader(dset_train, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(dset_test, batch_size=32, shuffle=True)


# Create an instance of the neural network
model = SimpleNet(input_dim, hidden_dim, hidden_dim2, hidden_dim3, output_dim)

trainer = pl.Trainer(max_epochs=25, accelerator='cpu')
trainer.fit(model, train_loader)

for k, l in {'train': train_loader, 'test': test_loader}.items():
    tupples = trainer.predict(model, dataloaders=l)

    preds = []
    ys = []
    for t in tupples:
        preds += [v[0] for v in t[0].tolist()]
        ys += [v[0] for v in t[1].tolist()]

    plt.scatter(ys, preds)
    plt.xlabel('y_'+k)
    plt.ylabel('Predictions')
    plt.title('Predictions vs. y_'+k+' '+str(np.corrcoef(ys, preds)[0, 1]))
    plt.show()

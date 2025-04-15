# -*- coding: utf-8 -*-
"""
1. Imports & device configuration
2. Data loading & preparation
3. DataLoader creation
4. Model definition (CNN architecture)
5. Hyperparameter and model instantiation
6. Training function definition and model training
"""

# 1. Imports & Device Setup
import os
import pickle
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import torch.optim as optim

# Set device (GPU if available; else CPU)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 2. Data Loading & Preparation
path_output = r'C:\Users\ishit\Downloads'
with open(os.path.join(path_output, 'EEGSpecWindows'), 'rb') as file:
    data2 = pickle.load(file)

badApples = [101,203,305,407,509,611,713,815,917,1014]   
eegData = [x[0] for x in data2]
specData = [x[1] for x in data2]

for i in sorted(badApples, reverse=True):
    if i < len(eegData):
        del eegData[i]
        del specData[i]
        
        
badOranges = [5, 6, 10, 12, 19, 20, 28, 35, 43, 46, 50, 51, 56, 57, 66, 67, 69, 71, 75, 81, 82, 89, 91, 96, 97, 103]  
eegData = [df.drop(df.columns[badOranges], axis=1, errors='ignore') for df in eegData]

random.shuffle(eegData)
random.shuffle(specData)

xTrain = np.array(eegData[0:800])
yTrain = np.array(specData[0:800])
xVal   = np.array(eegData[800:1016])
yVal   = np.array(specData[800:1016])



Xtrain = torch.from_numpy(xTrain).float()
Ytrain = torch.from_numpy(yTrain).float()
Xval   = torch.from_numpy(xVal).float()
Yval   = torch.from_numpy(yVal).float()

trainDataset = TensorDataset(Xtrain, Ytrain)
valDataset   = TensorDataset(Xval, Yval)

# 3. DataLoader Creation
BATCH_SIZE = 32
train_loader = DataLoader(trainDataset, batch_size=BATCH_SIZE, shuffle=False)
val_loader   = DataLoader(valDataset, batch_size=BATCH_SIZE, shuffle=False)

# Print shape for the first batch
for i, (x_batch, y_batch) in enumerate(train_loader):
    if i == 0:
        print(f"Batch {i}: x shape = {x_batch.shape}, y shape = {y_batch.shape}")
        break

# 4. Model Definition – CNN
class CNN(nn.Module):
    def __init__(self, input_size, num_classes):
        super(CNN, self).__init__()
        self.cnn = nn.Sequential(
            nn.Conv1d(in_channels=input_size, out_channels=33, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),  # [batch, 33, 148]
            nn.Conv1d(in_channels=33, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2)   # [batch, 128, 74]
        )
        self.fc = nn.Linear(128 * 74, num_classes)

    def forward(self, x):
        out = self.cnn(x)                # [batch, 128, 74]
        out = out.view(out.size(0), -1)  # Flatten to [batch, 9472]
        out = self.fc(out)               # [batch, num_classes]
        return out
#%%
# 5. Hyperparameters & Model Instantiation
INPUT_SIZE  = xTrain.shape[2]                 # e.g., 16
OUTPUT_SIZE = y_batch.shape[1]*y_batch.shape[2]            # number of classes

print("Hyperparameters:", INPUT_SIZE, OUTPUT_SIZE)

model = CNN(input_size=INPUT_SIZE, num_classes=OUTPUT_SIZE).to(DEVICE)

# 6. Training Function & Execution
def train_model(model, train_loader, epochs):
    global train_losses
    criterion = nn.MSELoss()
    print("Training model:", model.__class__.__name__)
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    train_losses = []
    for epoch in range(epochs):
        print("EPOCH:")
        print(epoch)
        for i, (x, y) in enumerate(train_loader):
            x = x.permute(0, 2, 1).to(DEVICE)  # [batch, channels, seq_len]
            y = y.to(DEVICE)
            optimizer.zero_grad()
            y_pred = model(x)
            y = y.view(y.size(0), -1)
            loss = criterion(y_pred, y)
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())
            
            if (i + 1) % 100 == 0:
                print(f'Epoch [{epoch+1}/{epochs}], Step [{i+1}/{len(train_loader)}], Train Loss: {loss.item():.4f}')
        print("LOSS:")
        print (np.average(train_losses))
    return train_losses

# Run training
EPOCHS = 20

loss_history = train_model(model, train_loader, EPOCHS)


for i, (x, y) in enumerate(train_loader):
        y = y.view(y.size(0), -1)  # Flatten to [batch, 9472]
        print(y.shape)
    


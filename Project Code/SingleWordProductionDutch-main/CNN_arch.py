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
path_output = r'C:\Users\Arman\Desktop\ECE-542-Project-main' 
with open(os.path.join(path_output, 'EEGSpecWindows'), 'rb') as file:
    data2 = pickle.load(file)

badApples = [101,203,305,407,509,611,713,815,917,1014]   
eegData = [x[0] for x in data2]
specData = [x[1] for x in data2]

for i in sorted(badApples, reverse=True):
    if i < len(eegData):
        del eegData[i]
        del specData[i]
        

eegData = [df.drop(df.columns[0], axis=1, errors='ignore') for df in eegData]

random.shuffle(eegData)
random.shuffle(specData)


xTrain = np.array(eegData[0:800])
yTrain = np.array(specData[0:800])
xVal   = np.array(eegData[800:1016])
yVal   = np.array(specData[800:1016])

noiseTrain = np.random.normal(0,0.1,size=xTrain.shape)
noiseVal = np.random.normal(0,0.1,size=xVal.shape)

xTrain = xTrain+ noiseTrain
xVal = xVal + noiseVal



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
        x_batch = x_batch.unsqueeze(1)
        y_batch = y_batch.unsqueeze(1)
        print(f"Batch {i}: x shape = {x_batch.shape}, y shape = {y_batch.shape}")
        break



print(f"Batch {i}: x shape = {x_batch.shape}, y shape = {y_batch.shape}")

# 4. Model Definition – CNN
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.MaxPool2d(kernel_size=2, stride=2),  # [batch, 33, 148]
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.MaxPool2d(kernel_size=2, stride=2),   # [batch, 128, 74]
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.MaxPool2d(kernel_size=2, stride=2),  # [batch, 128, 74]
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.MaxPool2d(kernel_size=2, stride=2),   # [batch, 128, 74]
        )
        
        self.fc = nn.Linear(27648,1)
        
    def forward(self, x):
        out = self.cnn(x)                # [batch, 128, 74]
        out = out.view(out.size(0), -1)  # Flatten to [batch, 9472]
        out = self.fc(out)               # [batch, num_classes]
        return out
#%%
# 5. Hyperparameters & Model Instantiation
INPUT_SIZE  = xTrain.shape[1]                 # e.g., 16
OUTPUT_SIZE = y_batch.shape[1]*y_batch.shape[2]            # number of classes

print("Hyperparameters:", INPUT_SIZE, OUTPUT_SIZE)

print("Input to model:", x_batch.shape)

model = CNN().to(DEVICE)

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
            x = x.unsqueeze(1).to(DEVICE)
            #x = x.permute(1, 0, 2, 3).to(DEVICE)  # [batch, channels, seq_len]
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
    


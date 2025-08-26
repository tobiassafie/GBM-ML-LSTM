# 2.2 - Trying a channel embedding w/ masked 

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler, MinMaxScaler

import sys
sys.path.append("C:/Users/tobys/Downloads/GBM-ML-main/GBM-ML-main")

'''
This section is copied from the original encode.py script.
This is all of our data loading and processing.
'''

# Process data
lcs = pd.read_csv('lcs.csv')
channels = ['n0', 'n1', 'n2', 'n3', 'n4', 'n5', 'n6', 'n7', 'n8', 'n9', 'na', 'nb', 'b0', 'b1']

# MASK --- instead of filling with noise, binary mask NaNs

# 1) Presence mask (1 = observed, 0 = missing)
for c in channels:
    lcs[c + "_mask"] = (~lcs[c].isna()).astype(float)

# 2) Per-channel standardization using only observed values
for c in channels:
    mu = lcs[c].mean(skipna=True)
    sig = lcs[c].std(skipna=True)
    # Standardize observed entries; leave NaNs as NaN
    lcs[c] = (lcs[c] - mu) / (sig + 1e-8)

# 3) Zero-impute *after* standardization (so 0 means "no signal provided")
for c in channels:
    lcs[c] = lcs[c].fillna(0.0)

time_series_list = []
mask_series_list = []
burst_ids = []

for burst, group in lcs.groupby('burst'):
    X = group[channels].values                      # [T, C]
    M = group[[c + "_mask" for c in channels]].values  # [T, C]
    time_series_list.append(torch.tensor(X, dtype=torch.float32))
    mask_series_list.append(torch.tensor(M, dtype=torch.float32))
    burst_ids.append(burst)

# Pad to common T with zeros for data and zeros for mask
time_series_list = nn.utils.rnn.pad_sequence(time_series_list, batch_first=True, padding_value=0.0)  # [B,T,C]
mask_series_list = nn.utils.rnn.pad_sequence(mask_series_list, batch_first=True, padding_value=0.0)  # [B,T,C]

# Convert to tensor (we already standardized per-channel pre-padding)
time_series_list = torch.tensor(time_series_list, dtype=torch.float32)


# Dataset Class
class GRBDataset(Dataset):
    def __init__(self, data, masks):
        self.data = data
        self.masks = masks
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        return self.data[idx], self.masks[idx]
    

'''
LSTM Autoencoder

Features:
- Bidirectional
- Attention
- Scheduler --> Reduce on Plateau
- Channel Embedding
- Channel Mask w/ stem
- 
'''

# Input stem - Allows the model to identify the difference between masked and nonmasked channels
class InputStem(nn.Module):
    def __init__(self, C):
        super().__init__()
        # Compress [x, mask] of size 2C back down to C
        self.proj = nn.Linear(2 * C, C)

    def forward(self, x, m):
        # Concatenate along last dim: [B,T,C] + [B,T,C] -> [B,T,2C]
        combined = torch.cat([x, m], dim=-1)
        return self.proj(combined)  # [B,T,C]

# Channel embedding -- allows the model to identify each channel for better learning
class ChannelEmbedding(nn.Module):
    def __init__(self, input_dim):
        super(ChannelEmbedding, self).__init__()
        self.channel_embedding = nn.Linear(input_dim, input_dim)

    def forward(self, x):
        # x: [B, T, C]
        B, T, C = x.shape
        channel_indices = torch.arange(C, device=x.device).float()  # [C]
        channel_indices = channel_indices.unsqueeze(0).unsqueeze(0).expand(B, T, C)  # [B, T, C]
        channel_embedded = self.channel_embedding(channel_indices)  # [B, T, model_dim]
        return x + channel_embedded


# Bidirectional LSTM Autoencoder Model w/ attention
class Encoder(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, latent_size, dropout):
        super().__init__()
        self.stem = InputStem(input_size)
        self.channel_embedding = ChannelEmbedding(input_size)
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout,
            bidirectional=True
        )
        self.attention = nn.Linear(hidden_size * 2, 1)
        self.fc_latent = nn.Linear(hidden_size * 2, latent_size)

    def forward(self, x, m):  # note: takes both x and mask
        x = self.stem(x, m)                # integrate mask → [B,T,C]
        x = self.channel_embedding(x)      # add channel embedding
        out, _ = self.lstm(x)
        attn_scores = self.attention(out)
        attn_weights = torch.softmax(attn_scores, 1)
        context = torch.sum(attn_weights * out, dim=1)
        latent = self.fc_latent(context)
        return latent, attn_weights

class Decoder(nn.Module):
    def __init__(self, latent_size, hidden_size, num_layers, output_size, seq_len):
        super().__init__()
        self.fc_expand = nn.Linear(latent_size, hidden_size * 2)
        self.lstm = nn.LSTM(
            input_size=hidden_size * 2,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True
        )
        self.fc_out = nn.Linear(hidden_size * 2, output_size)
        self.seq_len = seq_len

    def forward(self, latent):
        # Expand latent vector to all timesteps
        repeated = self.fc_expand(latent).unsqueeze(1).repeat(1, self.seq_len, 1)
        
        output, _ = self.lstm(repeated)     # [batch, time, hidden_size*2]
        output = self.fc_out(output)        # [batch, time, output_size]
        return output


class BiLSTMAutoencoder(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, latent_size, seq_len, dropout):
        super().__init__()
        self.encoder = Encoder(input_size, hidden_size, num_layers, latent_size, dropout)
        self.decoder = Decoder(latent_size, hidden_size, num_layers, input_size, seq_len)

    def forward(self, x, m):
        latent, attn_weights = self.encoder(x, m)
        reconstructed = self.decoder(latent)
        return reconstructed, latent, attn_weights
    

# Standard Training and Implementation

# Parameters
input_dim       = 14       # Number of detectors (features per timestep)
hidden_dim      = 16       # LSTM hidden state size
latent_dim      = 64       # Size of latent representation (embedding)
num_layers      = 2        # Number of LSTM layers
dropout         = 0.4      # Dropout between LSTM layers
batch_size      = 16       # Number of GRBs per batch
num_epochs      = 20       # Training epochs
learning_rate   = 0.00012  # Optimizer learning rate
sequence_length = np.shape(time_series_list)[1]  # Timesteps per GRB


model = BiLSTMAutoencoder(
    input_dim,
    hidden_dim,
    num_layers,
    latent_dim,
    sequence_length,
    dropout
)

# Define the loss function and optimizer and scheduler
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=3, factor=0.5)

# Get data
dataset   = GRBDataset(time_series_list, mask_series_list)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Training loop
for epoch in range(num_epochs):
    for i, (batch, batch_mask) in enumerate(dataloader):
        batch, batch_mask = batch.float(), batch_mask.float()
        optimizer.zero_grad()
        reconstructed, _, _ = model(batch, batch_mask)

        # Masked loss
        se = (reconstructed - batch) ** 2 * batch_mask
        loss = se.sum() / (batch_mask.sum() + 1e-8)

        loss.backward()
        optimizer.step()
        scheduler.step(loss)


        # Print the final batch loss for this epoch
        if i == len(dataloader) - 1:
            print(f"Epoch {epoch+1}, Final batch loss: {loss.item():.4f}")
    
# Extract latent features / Inference
model.eval()
latent_feats = []
with torch.no_grad():
    for batch, batch_mask in dataloader:
        _, latent, _ = model(batch, batch_mask)
        latent_feats.append(latent.numpy())

latent_feats = np.concatenate(latent_feats, axis=0)
np.save("latent_feats.npy", latent_feats)
np.save("burst_list.npy", np.array(burst_ids))
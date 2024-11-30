import pandas as pd
import matplotlib.pyplot as plt
import math

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

lcs = pd.read_csv('lcs.csv')
channels = ['n0', 'n1', 'n2', 'n3', 'n4', 'n5', 'n6', 'n7', 'n8', 'n9', 'na', 'nb', 'b1', 'b2']

from sklearn.preprocessing import MinMaxScaler, StandardScaler

def rescale_data(df, channels, scaler_type='minmax'):
    if scaler_type == 'minmax':
        scaler = MinMaxScaler()
    elif scaler_type == 'standard':
        scaler = StandardScaler()
    else:
        raise ValueError("scaler_type must be 'minmax' or 'standard'")
    
    df[channels] = scaler.fit_transform(df[channels])
    return df

lcs = rescale_data(lcs, channels, scaler_type='minmax')

# Fill missing channels with zeros
for channel in channels:
  lcs[channel].fillna(0.0, inplace=True)

time_series_list = []
grouped = lcs.groupby('burst')
for burst, group in grouped:
    time_series_data = group[channels].values
    time_series_tensor = torch.tensor(time_series_data, dtype=torch.float32)
    time_series_list.append(time_series_tensor)

class TimeSeriesDataset(Dataset):
    def __init__(self, time_series_list):
        self.time_series_list = time_series_list

    def __len__(self):
        return len(self.time_series_list)

    def __getitem__(self, idx):
        return self.time_series_list[idx]

# Pad sequences to the same length
def collate_fn(batch):
    batch = nn.utils.rnn.pad_sequence(batch, batch_first=True, padding_value=0.0)
    return batch

# Define the Positional Encoding Class
class PositionalEncoding(nn.Module):
    def __init__(self, model_dim, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, model_dim)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, model_dim, 2).float() * (-math.log(10000.0) / model_dim))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return x

# Define the Transformer Encoder
class TransformerEncoder(nn.Module):
    def __init__(self, input_dim, model_dim, num_heads, num_layers, dropout=0.1):
        super(TransformerEncoder, self).__init__()
        self.embedding = nn.Linear(input_dim, model_dim)
        self.positional_encoding = PositionalEncoding(model_dim)
        encoder_layer = nn.TransformerEncoderLayer(d_model=model_dim, nhead=num_heads, dropout=dropout)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

    def forward(self, src):
        src = self.embedding(src)
        src = self.positional_encoding(src)
        output = self.transformer_encoder(src)
        return output

# Define the Transformer Decoder
class TransformerDecoder(nn.Module):
    def __init__(self, model_dim, output_dim, num_heads, num_layers, dropout=0.1):
        super(TransformerDecoder, self).__init__()
        decoder_layer = nn.TransformerDecoderLayer(d_model=model_dim, nhead=num_heads, dropout=dropout)
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
        self.output_layer = nn.Linear(model_dim, output_dim)

    def forward(self, tgt, memory):
        output = self.transformer_decoder(tgt, memory)
        output = self.output_layer(output)
        return output

# Define the Autoencoder
class TransformerAutoencoder(nn.Module):
    def __init__(self, input_dim, model_dim, num_heads, num_layers, dropout=0.1):
        super(TransformerAutoencoder, self).__init__()
        self.encoder = TransformerEncoder(input_dim, model_dim, num_heads, num_layers, dropout)
        self.decoder = TransformerDecoder(model_dim, input_dim, num_heads, num_layers, dropout)

    def forward(self, src):
        memory = self.encoder(src)
        tgt = self.encoder.embedding(src)
        tgt = self.encoder.positional_encoding(tgt)
        output = self.decoder(tgt, memory)
        return output, memory

# Parameters
input_dim = 14
model_dim = 32
num_heads = 4
num_layers = 2
batch_size = 16
num_epochs = 20
learning_rate = 0.001

# Initialize the autoencoder
autoencoder = TransformerAutoencoder(input_dim, model_dim, num_heads, num_layers)

# Create the dataset and dataloader
dataset = TimeSeriesDataset(time_series_list)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)

# Define the loss function and optimizer
criterion = nn.MSELoss()
optimizer = optim.Adam(autoencoder.parameters(), lr=learning_rate)

# Training loop
for epoch in range(num_epochs):
    for batch in dataloader:
        # Original shape of batch: (batch_size, sequence_length, input_dim)
        # Permute to shape: (sequence_length, batch_size, input_dim)
        batch = batch.permute(1, 0, 2)
        reconstructed, latent = autoencoder(batch)
        loss = criterion(reconstructed, batch)

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

latent_feats = []
for batch in dataloader:
    batch = batch.permute(1, 0, 2)
    reconstructed, latent = autoencoder(batch)
    latent_feats.append(latent)

latent_feats = torch.cat(latent_feats, dim=1)

torch.save(latent_feats, 'latent_feats.pt')


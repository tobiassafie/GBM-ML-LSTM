import numpy as np
import pandas as pd
import math

from sklearn.preprocessing import StandardScaler, MinMaxScaler
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

lcs = pd.read_csv('lcs.csv')
channels = ['n0', 'n1', 'n2', 'n3', 'n4', 'n5', 'n6', 'n7', 'n8', 'n9', 'na', 'nb', 'b0', 'b1']

# Fill missing channels with noise
for channel in channels: 
    missing_indices = lcs[channel].isnull()  
    num_missing = missing_indices.sum()
    noise = np.random.normal(loc=lcs[channel].mean(), scale=lcs[channel].std(), size=num_missing)  
    lcs.loc[missing_indices, channel] = noise   

time_series_list = []
burst_ids = []
grouped = lcs.groupby('burst')
for burst, group in grouped:
    time_series_data = group[channels].values
    time_series_tensor = torch.tensor(time_series_data, dtype=torch.float32)
    time_series_list.append(time_series_tensor)
    burst_ids.append(burst)

# Padding with zeros
time_series_list = nn.utils.rnn.pad_sequence(time_series_list, batch_first=True, padding_value=0.0)

# Normalize the light curves
scaler = StandardScaler()
time_series_list_2d = time_series_list.reshape(time_series_list.shape[0], -1)
time_series_list_2d = scaler.fit_transform(time_series_list_2d)
time_series_list = time_series_list_2d.reshape(time_series_list.shape)

class TimeSeriesDataset(Dataset):
    def __init__(self, time_series_list, burst_ids):
        self.time_series_list = time_series_list
        self.burst_ids = burst_ids

    def __len__(self):
        return len(self.time_series_list)

    def __getitem__(self, idx):
        return self.time_series_list[idx], self.burst_ids[idx]

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
        return x + self.pe[:x.size(0), :]

# Define the Channel Embedding Class
class ChannelEmbedding(nn.Module):
    def __init__(self, input_dim, model_dim):
        super(ChannelEmbedding, self).__init__()
        self.channel_embedding = nn.Linear(input_dim, model_dim)

    def forward(self, x):
        channel_indices = torch.arange(input_dim).unsqueeze(0).expand(x.size(0), x.size(1), -1) 
        channel_embedded = self.channel_embedding(channel_indices.float())   
        return x + channel_embedded    

# Define the Transformer Encoder
class TransformerEncoder(nn.Module):
    def __init__(self, input_dim, model_dim, num_heads, num_layers, dropout=0.1):
        super(TransformerEncoder, self).__init__()
        self.embedding = nn.Sequential(
            nn.Linear(input_dim, model_dim),
            nn.ReLU()  
        )
        self.channel_embedding = ChannelEmbedding(input_dim, model_dim)
        self.positional_encoding = PositionalEncoding(model_dim)
        encoder_layer = nn.TransformerEncoderLayer(d_model=model_dim, nhead=num_heads, dropout=dropout)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

    def forward(self, src):
        src = self.embedding(src) 
        src = self.channel_embedding(src)        
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

# Define the Compression Neural Network
class IntermediateNetwork(nn.Module):
    def __init__(self, latent_dim, reduced_dim, sequence_length):
        super(IntermediateNetwork, self).__init__()
        self.reduce = nn.Sequential(
            nn.Linear(sequence_length * latent_dim, reduced_dim),
            nn.ReLU()  
        )
        self.expand = nn.Sequential(
            nn.Linear(reduced_dim, sequence_length * latent_dim),
            nn.ReLU() 
        )

    def forward(self, x):
        # Reduce dimensionality
        org_shape = x.size()
        x = x.reshape(x.size(1), -1)  # Reshape to (batch_size, sequence_length * latent_dim)
        reduced_x = self.reduce(x)

        # Expand dimensionality
        expanded_x = self.expand(reduced_x)
        expanded_x = expanded_x.reshape(org_shape)  # Reshape back

        return expanded_x, reduced_x 
    
# Define the Autoencoder
class TransformerAutoencoder(nn.Module):
    def __init__(self, input_dim, model_dim, num_heads, num_layers, sequence_length, reduced_dim, dropout=0.1):  
        super(TransformerAutoencoder, self).__init__()
        self.encoder = TransformerEncoder(input_dim, model_dim, num_heads, num_layers, dropout)
        self.intermediate_network = IntermediateNetwork(model_dim, reduced_dim, sequence_length)
        self.decoder = TransformerDecoder(model_dim, input_dim, num_heads, num_layers, dropout)  

    def forward(self, src):
        memory = self.encoder(src)
        expanded_latent, reduced_latent = self.intermediate_network(memory)  
        tgt = self.encoder.embedding(src)  
        tgt = self.encoder.positional_encoding(tgt)  
        output = self.decoder(tgt, expanded_latent)  
        return output, reduced_latent  
    
# Parameters
input_dim = 14
model_dim = 32
num_heads = 4
num_layers = 2
reduced_dim = 1024
batch_size = 16
num_epochs = 50
learning_rate = 0.0001
dropout = 0.1
sequence_length = np.shape(time_series_list)[1]
# Define the loss function and optimizer
criterion = nn.MSELoss()
optimizer = optim.Adam(autoencoder.parameters(), lr=learning_rate)

dataset = TimeSeriesDataset(time_series_list, burst_ids)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Initialize the autoencoder
autoencoder = TransformerAutoencoder(input_dim, model_dim, num_heads, num_layers, sequence_length, reduced_dim, dropout=dropout)

# Training 
for epoch in range(num_epochs):
    for batch_idx, (batch, bursts) in enumerate(dataloader):
        # Original shape of batch: (batch_size, sequence_length, input_dim)
        # Permute to shape: (sequence_length, batch_size, input_dim)
        batch = batch.permute(1, 0, 2).float()
        reconstructed, latent = autoencoder(batch)
        loss = criterion(reconstructed, batch)

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

torch.save(autoencoder, 'autoencoder.pt')
#torch.load('autoencoder.pt', weights_only=False, map_location=torch.device('cpu'))

# Inference 
latent_feats = np.empty((0, reduced_dim))
burst_list = np.empty((0,))
for batch_idx, (batch, bursts) in enumerate(dataloader):
    batch = batch.permute(1, 0, 2).float()
    reconstructed, latent = autoencoder(batch)
    latent_feats = np.concatenate([latent_feats, latent.cpu().detach().numpy()], axis=0)
    burst_list = np.concatenate([burst_list, bursts])

latent_feats = latent_feats.reshape(-1, latent_feats.shape[1])
np.save('latent_feats.npy', latent_feats)
np.save('burst_list.npy', burst_list)


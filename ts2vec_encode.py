import numpy as np
import pandas as pd
import math
import random
from typing import Tuple

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler

# ---------------------------
# PREPROCESSING
# ---------------------------

lcs = pd.read_csv('lcs.csv')
channels = ['n0','n1','n2','n3','n4','n5','n6','n7','n8','n9','na','nb','b0','b1']

# Fill missing channels with noise (same approach)
for ch in channels:
    missing = lcs[ch].isnull()
    num_missing = missing.sum()
    if num_missing > 0:
        noise = np.random.normal(loc=lcs[ch].mean(), scale=lcs[ch].std(), size=num_missing)
        lcs.loc[missing, ch] = noise

# Group by burst → list of variable-length tensors
time_series_list = []
burst_ids = []
for burst, grp in lcs.groupby('burst'):
    arr = grp[channels].values  # [Ti, 14]
    time_series_list.append(torch.tensor(arr, dtype=torch.float32))
    burst_ids.append(burst)

# Pad to common length with zeros
time_series_padded = nn.utils.rnn.pad_sequence(time_series_list, batch_first=True, padding_value=0.0)  # [N, T, C]

# Standardize *after* padding, on the flattened view (identical pattern)
scaler = StandardScaler()
N, T, C = time_series_padded.shape
flat = time_series_padded.reshape(N, -1).numpy()
flat = scaler.fit_transform(flat)
time_series_padded = torch.tensor(flat.reshape(N, T, C), dtype=torch.float32)

# ---------------------------
# DATASET + AUGMENTATIONS
# ---------------------------

class GRBDataset(Dataset):
    def __init__(self, data: torch.Tensor):
        self.data = data  # [N, T, C]
    def __len__(self): return self.data.shape[0]
    def __getitem__(self, idx): return self.data[idx]  # [T, C]

def random_time_mask(x: torch.Tensor, max_frac: float = 0.15) -> torch.Tensor:
    """Zero out a random contiguous segment (≤ max_frac of T)."""
    T = x.shape[0]
    if T < 4: return x
    seg_len = max(1, int(T * random.uniform(0.05, max_frac)))
    start = random.randint(0, max(0, T - seg_len))
    x_aug = x.clone()
    x_aug[start:start+seg_len] = 0.0
    return x_aug

def jitter(x: torch.Tensor, sigma: float = 0.02) -> torch.Tensor:
    """Add small Gaussian jitter."""
    return x + torch.randn_like(x) * sigma

def random_crop(x: torch.Tensor, min_frac: float = 0.5) -> torch.Tensor:
    """Random crop then pad back to original T with zeros at the end."""
    T, C = x.shape
    keep = max(4, int(T * random.uniform(min_frac, 1.0)))
    start = random.randint(0, T - keep)
    crop = x[start:start+keep]
    if keep < T:
        pad = torch.zeros(T - keep, C, dtype=x.dtype, device=x.device)
        crop = torch.cat([crop, pad], dim=0)
    return crop

def two_views(batch: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Produce two stochastic augmentations for SimCLR-style contrast."""
    # batch: [B, T, C]
    views1, views2 = [], []
    for x in batch:
        v1 = random_crop(jitter(random_time_mask(x)))
        v2 = random_crop(jitter(random_time_mask(x)))
        views1.append(v1)
        views2.append(v2)
    return torch.stack(views1, 0), torch.stack(views2, 0)  # [B, T, C] each

# ---------------------------
# TCN BACKBONE
# ---------------------------

class TemporalBlock(nn.Module):
    def __init__(self, in_ch, out_ch, kernel=3, dilation=1, dropout=0.1):
        super().__init__()
        padding = (kernel - 1) * dilation
        self.net = nn.Sequential(
            nn.Conv1d(in_ch, out_ch, kernel, padding=padding, dilation=dilation),
            nn.ReLU(),
            nn.BatchNorm1d(out_ch),
            nn.Dropout(dropout),
            nn.Conv1d(out_ch, out_ch, kernel, padding=padding, dilation=dilation),
            nn.ReLU(),
            nn.BatchNorm1d(out_ch),
            nn.Dropout(dropout),
        )
        self.downsample = nn.Conv1d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x):  # x: [B, C, T]
        out = self.net(x)
        # causal crop to keep length T (remove padding on left)
        crop = out[..., :x.size(-1)]
        return crop + self.downsample(x)

class TCNEncoder(nn.Module):
    def __init__(self, in_dim=14, hid=64, layers=4, dropout=0.1, out_dim=128):
        super().__init__()
        blocks = []
        ch_in = in_dim
        for i in range(layers):
            ch_out = hid
            blocks.append(TemporalBlock(ch_in, ch_out, kernel=3, dilation=2**i, dropout=dropout))
            ch_in = ch_out
        self.tcn = nn.Sequential(*blocks)
        self.proj = nn.Linear(hid, out_dim)

    def forward(self, x):  # x: [B, T, C]
        x = x.permute(0, 2, 1)        # [B, C, T]
        h = self.tcn(x)               # [B, H, T]
        h = h.permute(0, 2, 1)        # [B, T, H]
        z = self.proj(h)              # [B, T, D]
        return z

class MeanPoolHead(nn.Module):
    def forward(self, z):             # z: [B, T, D]
        return z.mean(dim=1)          # [B, D]

# ---------------------------
# NT-Xent (InfoNCE) LOSS
# ---------------------------

def nt_xent(z1, z2, tau=0.2):
    """Normalized temperature-scaled cross entropy (SimCLR)."""
    # z1,z2: [B, D]
    B, D = z1.shape
    z1 = nn.functional.normalize(z1, dim=1)
    z2 = nn.functional.normalize(z2, dim=1)
    reps = torch.cat([z1, z2], dim=0)         # [2B, D]
    logits = reps @ reps.t() / tau            # [2B, 2B]
    mask = torch.eye(2*B, dtype=torch.bool, device=logits.device)
    logits = logits.masked_fill(mask, -1e9)

    targets = torch.arange(B, device=logits.device)
    targets = torch.cat([targets + B, targets], dim=0)   # positives across views

    loss = nn.functional.cross_entropy(logits, targets)
    return loss

# ---------------------------
# TRAIN
# ---------------------------

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Hyperparameters (feel free to sweep)
batch_size      = 32
num_epochs      = 30
learning_rate   = 1e-3
tcn_hidden      = 64
tcn_layers      = 4
proj_dim        = 128
dropout         = 0.1
temperature     = 0.2

dataset = GRBDataset(time_series_padded)  # [N, T, C]
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)

encoder = TCNEncoder(in_dim=channels.__len__(), hid=tcn_hidden, layers=tcn_layers,
                     dropout=dropout, out_dim=proj_dim).to(device)
pooler = MeanPoolHead().to(device)

optimizer = optim.Adam(encoder.parameters(), lr=learning_rate)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=3, factor=0.5)

encoder.train()
for epoch in range(num_epochs):
    running = 0.0
    for batch in dataloader:
        batch = batch.to(device)  # [B, T, C]
        v1, v2 = two_views(batch) # augment
        v1, v2 = v1.to(device), v2.to(device)

        z1 = pooler(encoder(v1))  # [B, D]
        z2 = pooler(encoder(v2))  # [B, D]
        loss = nt_xent(z1, z2, tau=temperature)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running += loss.item()

    epoch_loss = running / len(dataloader)
    scheduler.step(epoch_loss)
    print(f"Epoch {epoch+1:02d}  |  contrastive loss: {epoch_loss:.4f}")

# ---------------------------
# EXTRACT & SAVE EMBEDDINGS
# ---------------------------

encoder.eval()
with torch.no_grad():
    # Use *unaugmented* sequences, mean-pool over time for burst-level embedding
    all_feats = []
    for i in range(0, len(dataset), batch_size):
        batch = time_series_padded[i:i+batch_size].to(device)  # [b, T, C]
        zt = encoder(batch)            # [b, T, D]
        zv = pooler(zt)                # [b, D]
        all_feats.append(zv.cpu().numpy())

latent_feats = np.concatenate(all_feats, axis=0)   # [N, D]
burst_list = np.array(burst_ids)

np.save("latent_feats.npy", latent_feats)
np.save("burst_list.npy", burst_list)
print("Saved latent_feats.npy and burst_list.npy")

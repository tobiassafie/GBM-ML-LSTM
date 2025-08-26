#!/usr/bin/env python3
# embed_by_t90.py
# UMAP (paper-style params) + PCA, colored by log10(T90)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# ---------- paths ----------
LATENT_PATH = "latent_feats.npy"
BURST_LIST_PATH = "burst_list.npy"
CLUSTERED_CSV = "clustered_with_durations.csv"
OUTDIR = Path("embeds")

# ---------- load ----------
latent = np.load(LATENT_PATH)
burst_ids = np.load(BURST_LIST_PATH, allow_pickle=True).astype(int)
dur_df = pd.read_csv(CLUSTERED_CSV)[["burst_id", "duration"]]

df = pd.DataFrame(latent)
df["burst_id"] = burst_ids
df = df.merge(dur_df, on="burst_id", how="left")

# drop bad durations and compute log10(T90)
df = df[(df["duration"] > 0) & np.isfinite(df["duration"])]
logt90 = np.log10(df["duration"].values)

# feature matrix
X = df.drop(columns=["burst_id", "duration"]).values

# ---------- scale features (important for UMAP/PCA) ----------
X = StandardScaler().fit_transform(X)

OUTDIR.mkdir(parents=True, exist_ok=True)

# ---------- PCA baseline (2D) ----------
pca = PCA(n_components=2, random_state=42)
X_pca = pca.fit_transform(X)
pd.DataFrame(X_pca, columns=["pca1", "pca2"]).to_csv(OUTDIR / "pca_2d.csv", index=False)

plt.figure(figsize=(7.2, 6))
sc = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=logt90, s=8, cmap="turbo")
plt.xlabel("PCA 1"); plt.ylabel("PCA 2")
plt.title("PCA (scaled latent) — colored by log10(T90)")
cbar = plt.colorbar(sc, label="log10(T90 [s])")
plt.tight_layout()
plt.savefig(OUTDIR / "pca_by_t90.png", dpi=220)
plt.close()

# ---------- UMAP (paper-like params) ----------
import umap

def run_umap(n_components):
    reducer = umap.UMAP(
        n_neighbors=30,
        min_dist=0.0,
        n_components=n_components,
        metric="euclidean",
        local_connectivity=0.5,
        n_epochs=1000,
        learning_rate=1e-3,
        random_state=42,
    )
    return reducer.fit_transform(X)

# UMAP 2D
U2 = run_umap(2)
pd.DataFrame(U2, columns=["umap1", "umap2"]).to_csv(OUTDIR / "umap_2d.csv", index=False)

plt.figure(figsize=(7.2, 6))
sc = plt.scatter(U2[:, 0], U2[:, 1], c=logt90, s=8, cmap="turbo")
plt.xlabel("UMAP-1"); plt.ylabel("UMAP-2")
plt.title("UMAP 2D (scaled latent) — colored by log10(T90)")
plt.colorbar(sc, label="log10(T90 [s])")
plt.tight_layout()
plt.savefig(OUTDIR / "umap2d_by_t90.png", dpi=220)
plt.close()

# UMAP 3D + multi-view panel (like the paper)
U3 = run_umap(3)
pd.DataFrame(U3, columns=["umap1", "umap2", "umap3"]).to_csv(OUTDIR / "umap_3d.csv", index=False)

from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

fig = plt.figure(figsize=(14, 9), constrained_layout=True)
gs = fig.add_gridspec(3, 2, width_ratios=[1, 1])

# shared colorbar setup
norm = Normalize(vmin=np.nanpercentile(logt90, 1), vmax=np.nanpercentile(logt90, 99))
cmap = plt.get_cmap("turbo")
sm = ScalarMappable(norm=norm, cmap=cmap)

def scatter_3d(ax, elev, azim, title):
    ax.view_init(elev=elev, azim=azim)
    ax.scatter(U3[:, 0], U3[:, 1], U3[:, 2], c=sm.to_rgba(logt90), s=6, linewidths=0)
    ax.set_xlabel("X"); ax.set_ylabel("Y"); ax.set_zlabel("Z")
    ax.set_title(title, pad=10)

# three 3D views on the left
angles = [(25, 35), (10, 125), (5, 205)]
for i, (elev, azim) in enumerate(angles):
    ax = fig.add_subplot(gs[i, 0], projection="3d")
    scatter_3d(ax, elev, azim, f"UMAP 3D — view {i+1}")

# 2D UMAP on the right spanning rows
ax2 = fig.add_subplot(gs[:, 1])
ax2.scatter(U2[:, 0], U2[:, 1], c=sm.to_rgba(logt90), s=6, linewidths=0)
ax2.set_xlabel("UMAP-1"); ax2.set_ylabel("UMAP-2")
ax2.set_title("UMAP 2D")

# shared colorbar
cbar = fig.colorbar(sm, ax=[fig.axes[0], fig.axes[1], fig.axes[2], ax2], fraction=0.046, pad=0.03)
cbar.set_label("log10(T90 [s])")

fig.suptitle("GRB Embeddings colored by log10(T90)", y=0.995)
fig.savefig(OUTDIR / "umap_by_t90_panel.png", dpi=240)
plt.close()

print("Saved files in:", OUTDIR)
print(" - pca_by_t90.png")
print(" - umap2d_by_t90.png")
print(" - umap_by_t90_panel.png")
print(" - pca_2d.csv, umap_2d.csv, umap_3d.csv")
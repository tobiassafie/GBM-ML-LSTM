from sklearn.mixture import BayesianGaussianMixture
from sklearn.manifold import TSNE
from sklearn.ensemble import IsolationForest

import torch

import matplotlib.pyplot as plt

latent_feats = torch.load('latent_feats.pt')
latent_feats_np = latent_feats.numpy()

bgm = BayesianGaussianMixture(n_components=10).fit(latent_latent_feats_np)
cluster_assignments = bgm.predict(latent_feats_np)

# Apply t-SNE for dimensionality reduction
tsne = TSNE(n_components=2, random_state=42)
latent_feats_2d = tsne.fit_transform(latent_feats_np)

# Fit Isolation Forest to find outliers
iso_forest = IsolationForest(contamination=0.1, random_state=42)
outlier_labels = iso_forest.fit_predict(latent_feats_np)

# Visualize clusters and outliers
plt.figure(figsize=(10, 8))
plt.scatter(latent_feats_2d[:, 0], latent_feats_2d[:, 1], c=cluster_assignments, cmap='viridis', s=50, label='Clusters')
outliers = latent_feats_2d[outlier_labels == -1]
plt.scatter(outliers[:, 0], outliers[:, 1], c='red', s=50, label='Outliers')
plt.colorbar()
plt.title('Clusters and Outliers in Latent Space')
plt.xlabel('t-SNE Component 1')
plt.ylabel('t-SNE Component 2')
plt.legend()
plt.show()

'''
sed = pd.read_csv('sample_energy_lc.txt', delimiter=' ')
sed = sed.T.reset_index().rename(columns = {'index':'wavelength', 0:'count'})
plt.plot(sed['wavelength'].astype('float'), sed['count'].astype('float'))
'''


import numpy as np 
import matplotlib.pyplot as plt
from sklearn.mixture import BayesianGaussianMixture, GaussianMixture
from sklearn.manifold import TSNE
from sklearn.ensemble import IsolationForest

latent_feats = np.load('latent_feats.npy')
burst_list = np.load('burst_list.npy')

# Fit Bayesian Gaussian Mixture Model
bgm = GaussianMixture(n_components=2).fit(latent_feats)
cluster_assignments = bgm.predict(latent_feats)

# Apply t-SNE for dimensionality reduction
tsne = TSNE(n_components=2, perplexity=10, metric='l1')
latent_feats_2d = tsne.fit_transform(latent_feats)

# Fit Isolation Forest to find outliers
iso_forest = IsolationForest()
outlier_labels = iso_forest.fit_predict(latent_feats)

# Visualize clusters and outliers
plt.figure(figsize=(10, 8))
for cluster in np.unique(cluster_assignments):
    cluster_points = latent_feats_2d[cluster_assignments == cluster]
    if not cluster_points.size == 0:
        plt.scatter(cluster_points[:, 0], cluster_points[:, 1], label=f'Cluster {cluster}', s=5)
outliers = latent_feats_2d[outlier_labels == -1]
plt.scatter(outliers[:, 0], outliers[:, 1], c='red', s=5, label='Outliers')
plt.title('Clusters and Outliers in Latent Space')
plt.xlabel('t-SNE Component 1')
plt.ylabel('t-SNE Component 2')
plt.legend()
plt.show()
plt.savefig('clusters_outliers.png')

# Identify and print burst IDs of outliers
outlier_burst_ids = burst_list[outlier_labels == -1]
for burst_id in outlier_burst_ids:
        print(f"{burst_id:.0f}")

# Print burst IDs of clusters
for cluster in range(1,2):
    cluster_burst_ids = burst_list[cluster_assignments == cluster]
    print(f"Burst IDs of Cluster {cluster}:")
    for burst_id in cluster_burst_ids:
        print(f"{burst_id:.0f}")        


latent = []
for batch in dataloader:
    reconstructed, latent = autoencoder(batch)
    latent.append(latent.detach().cpu())

latent = torch.cat(latent, dim=1)

'''
from sklearn.mixture import BayesianGaussianMixture
bgm = BayesianGaussianMixture(n_components=10, random_state=42).fit(latent)
bgm.means_

sed = pd.read_csv('sample_energy_lc.txt', delimiter=' ')
sed = sed.T.reset_index().rename(columns = {'index':'wavelength', 0:'count'})
plt.plot(sed['wavelength'].astype('float'), sed['count'].astype('float'))
'''


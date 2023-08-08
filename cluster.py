import pandas as pd
import matplotlib.pyplot as plt

sed = pd.read_csv('sample_energy_lc.txt', delimiter=' ')
sed = sed.T.reset_index().rename(columns = {'index':'wavelength', 0:'count'})

plt.plot(sed['wavelength'].astype('float'), sed['count'].astype('float'))
plt.show()

lc = pd.read_csv('sample_grb_lc.txt', delimiter=' ')
lc = lc.T.reset_index().rename(columns = {'index': 'time', 0: 'count'})

plt.plot(lc['time'].astype('float'), lc['count'].astype('float'))
plt.show()

# choose rep: best SED model in first or all time, count per channel
# reduce feature

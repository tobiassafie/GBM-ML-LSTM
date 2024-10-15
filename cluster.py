import pandas as pd
import matplotlib.pyplot as plt
import os, glob

props = pd.DataFrame()
for grb in os.listdir('Bursts'):
  if not glob.glob('Bursts/'+grb+'/lc_'+grb+'_*.txt'): print(grb)
  for lc_file in glob.glob('Bursts/'+grb+'/lc_'+grb+'_*.txt'):
    lc = pd.read_csv(lc_file, delimiter=' ')
    lc = lc.T.reset_index().rename(columns={'index': 'time', 0: 'count'})
    lc['time'] = lc['time'].astype('float')
    params = os.path.splitext(lc_file)[0].split('_')
    det = params[2]
    tstart = float(params[3])
    tdur = float(params[4])
    snr = float(params[5])
    props = pd.concat([props, pd.DataFrame([[grb, det, snr, lc['time'].max()-lc['time'].min(), tdur,
                                        lc['count'].min(), lc['count'].max(),
                                        lc[(lc['time'] > tstart) & (lc['time'] < tstart+tdur)]['count'].min(),
                                        lc[(lc['time'] > tstart) & (lc['time'] < tstart+tdur)]['count'].max(),
                                        ]], columns=['name', 'detector', 'snr', 'lc_duration', 'burst_duration',
                                                     'min_count', 'max_count', 'min_burst_count', 'max_burst_count'])
                     ])

pd.options.display.max_columns = None
props.describe()
props.groupby('name').ngroups #why 214 when 216 folders
props.groupby('name')['detector'].count()
props.groupby('name')['burst_duration'].unique().astype(float).idxmax() #longest burst
props.groupby('name')['burst_duration'].unique().astype(float).idxmin() #shortest burst
props.groupby('name')['max_burst_count'].max().idxmax() #brightest burst, any detector
props.groupby('name')['max_burst_count'].max().idxmin() #faintest burst, any detector
props.corr().abs()

len(os.listdir('Bursts'))
# how renorm
# feature dist and correlation

sed = pd.read_csv('sample_energy_lc.txt', delimiter=' ')
sed = sed.T.reset_index().rename(columns = {'index':'wavelength', 0:'count'})
plt.plot(sed['wavelength'].astype('float'), sed['count'].astype('float'))

# reduce feature
# use full time and energy decomposition?

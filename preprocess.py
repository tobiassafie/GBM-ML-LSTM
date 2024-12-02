import pandas as pd
import matplotlib.pyplot as plt
import os, glob
import math

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
props.groupby('name').ngroups 
props.groupby('name')['detector'].count()
props.groupby('name')['burst_duration'].unique().astype(float).idxmax() #longest burst
props.groupby('name')['burst_duration'].unique().astype(float).idxmin() #shortest burst
props.groupby('name')['max_burst_count'].max().idxmax() #brightest burst, any detector
props.groupby('name')['max_burst_count'].max().idxmin() #faintest burst, any detector
props.corr(numeric_only=True).abs() #relation between min burst count and max (burst) count. interesting

channels = ['n0', 'n1', 'n2', 'n3', 'n4', 'n5', 'n6', 'n7', 'n8', 'n9', 'na', 'nb', 'b0', 'b1']

lcs = pd.DataFrame()
for grb in os.listdir('Bursts'):
  if glob.glob('Bursts/'+grb+'/lc_'+grb+'_*.txt'): 
    lc = pd.DataFrame(columns=['burst']+channels)
    for lc_file in glob.glob('Bursts/'+grb+'/lc_'+grb+'_*.txt'):
      counts = pd.read_csv(lc_file, delimiter=' ')
      counts = counts.T.reset_index().rename(columns={'index': 'time', 0: 'count'})
      params = os.path.splitext(lc_file)[0].split('_')
      lc[params[2]] = counts['count'].astype('float')
    lc['burst'] = grb
    lcs = pd.concat([lcs, lc])

lcs.to_csv('lcs.csv', index=False)

'''
sed = pd.read_csv('sample_energy_lc.txt', delimiter=' ')
sed = sed.T.reset_index().rename(columns = {'index':'wavelength', 0:'count'})
plt.plot(sed['wavelength'].astype('float'), sed['count'].astype('float'))
'''

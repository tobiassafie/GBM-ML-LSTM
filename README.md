# GBM-ML

GBM_download : script to download GBM data

GBM_read_processed: script to read the saved GBM data

In the Burst folder, the lightcurves and the SED associated to each lightcurve can be found in the files

 `lc_{burst}_{det}_{t90_start}_{t90}_{SNR}.txt `

 `spec_{burst}_{det}_{t90_start}_{t90}_{SNR}.txt `

Where:

| key | Description |
| --- | ----------- |
| burst | name of the burst |
| det | detector ID. Na detectors from n1, n2, n3, .., na, nb, and BGO are b1, b2|
| t90_start | Time that defines the start of the burst [s] |
| t90 | T90 - duration of the burst [s] |
| SNR | Signal to Noise Ratio  |
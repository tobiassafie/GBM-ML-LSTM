# GBM-ML

GBM_download : script to download GBM data

GBM_read_processed: script to read the saved GBM data

In the Burst folder, each folder has the data from a particular burst (bnYYMMDD), within the folder the lightcurves and the SED associated to the detectors that have a >3sigma detection can be found in the files

 `lc_{burst}_{det}_{t90_start}_{t90}_{SNR}.txt `

 `spec_{burst}_{det}_{t90_start}_{t90}_{SNR}.txt `

Where:

| key | Description |
| --- | ----------- |
| burst | name of the burst |
| det | detector ID. NaI detectors from n1, n2, n3, .., na, nb, and BGO are b1, b2|
| t90_start | Time that defines the start of the burst [s] |
| t90 | T90 - duration of the burst [s] |
| SNR | Signal to Noise Ratio  |

In the collage_{burst}.png, the lc and SED of the detectors with >3sigma detections.
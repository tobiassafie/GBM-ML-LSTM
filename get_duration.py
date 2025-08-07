import os
import re
import pandas as pd

# Your local Bursts directory
bursts_dir = r'C:\Users\tobys\OneDrive\Desktop\GBM-ML-LSTM\Bursts'

burst_durations = {}

# Regex pattern for lc_ files with decimal numbers
pattern = r'^lc_(\d+)_\w+_[-\d.]+_([-\d.]+)_[-\d.]+\.txt$'

# Walk through all burst folders and files
for root, dirs, files in os.walk(bursts_dir):
    for file in files:
        if file.startswith('lc_'):
            match = re.match(pattern, file)
            if match:
                burst_id = match.group(1)
                t90 = float(match.group(2))  # Extracted duration
                burst_durations[burst_id] = max(t90, burst_durations.get(burst_id, 0.0))

# Convert to DataFrame
durations_df = pd.DataFrame(list(burst_durations.items()), columns=['burst_id', 'duration'])

# Save locally
output_path = os.path.join(bursts_dir, 'durations_from_filenames.csv')
durations_df.to_csv(output_path, index=False)

print(f'Saved {len(durations_df)} durations to {output_path}')

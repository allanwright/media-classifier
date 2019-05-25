#%%
import os
import pandas as pd

raw_data_path = 'data/raw'
raw_data = pd.DataFrame()

for x in os.listdir(raw_data_path):
    x_path = '%s/%s' % (raw_data_path, x)
    if os.path.isdir(x_path):
        for y in os.listdir(x_path):
            y_path = '%s/%s/%s' % (raw_data_path, x, y)
            if os.path.isfile(y_path):
                print('Processing %s' % y_path)
                series = pd.read_csv(y_path, sep='\t', squeeze=True)
                df = pd.DataFrame()
                df['name'] = series
                df['category'] = x
                raw_data = raw_data.append(df, ignore_index=True)

# Remove any commas from the name column before writing to csv
raw_data['name'] = raw_data['name'].str.replace(',', '')
raw_data.to_csv('data/interim/combined.csv', index=False)
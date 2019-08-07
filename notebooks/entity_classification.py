#%%
import numpy as np
import pandas as pd

df = pd.read_csv('data/interim/combined.csv')
print(df.head())

#%%
new_df = pd.DataFrame(df['name'].str.split().tolist(), index=df.index).stack()
print(new_df.shape)
print(new_df.head())

#%%
new_df.to_csv('data/interim/entity_classification.csv', index=True)

# Define entity classes by type
# SHARED
# na, file extension

# APP
# name, version

# MOVIE
# name, resolution, encoding, uploader

# MUSIC
# artist name, album name, track name, track number, bitrate

# TV
# show name, episode name, season number, episode number, resolution
# encoding, uploader

#%%

#%%
import numpy as np
import pandas as pd

df = pd.read_csv('data/interim/combined.csv')
print(df.head())

#%%
# Create a numpy array with a length equal to the number of words in the sentence
df['label'] = df['name'].apply(lambda x: np.zeros(len(x.split())))

# Define entity classes by type
# SHARED
# file extension
# na

# APP
# name
# version

# MOVIE
# name
# resolution
# encoding
# uploader

# MUSIC
# artist name
# album name
# track name
# track number
# bitrate

# TV
# show name
# episode name
# season number
# episode number
# resolution
# encoding
# uploader

#%%
df.to_csv('data/interim/entity_classification.csv', index=False)

#%%

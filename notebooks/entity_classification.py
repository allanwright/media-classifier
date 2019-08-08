#%%
import numpy as np
import pandas as pd

df = pd.read_csv('data/interim/combined.csv')

#%%
df = pd.DataFrame(df['name'].str.split().tolist(), index=df.index).stack()
df = df.reset_index([0, 1])
df.columns = ['index', 'pos', 'word']
df['entity'] = ''
print(df.head())

#%%
df.loc[df.word.str.contains('^s\d+e\d+$'), 'entity'] = 'season_episode'

#%%
df.to_csv('data/interim/entity_classification.csv', index=False)
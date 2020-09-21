'''Notebook for visualing training data.

'''

# %%
import pandas as pd

df = pd.read_csv('../data/interim/combined.csv')

# %%
print(df.head())
print(df.describe())

# %%
print(df['category'].value_counts())

#%% Import modules
import pandas as pd
import seaborn as sns

#%% Read dataset
df = pd.read_csv('data/interim/combined.csv')

#%% Display top x rows
print(df.head(20))

#%% Display category distribution
sns.countplot(x='category', data=df)

#%%
df.drop_duplicates(subset=['name', 'category'], inplace=True)

print(df.head(20))

#%%
print(df.loc[df['category'] == 'apps', 'name'].value_counts())
print(df.loc[df['category'] == 'music', 'name'].value_counts())
print(df.loc[df['category'] == 'movies', 'name'].value_counts())
print(df.loc[df['category'] == 'tv', 'name'].value_counts())
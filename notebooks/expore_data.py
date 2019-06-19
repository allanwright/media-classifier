#%% Import modules and read dataset
import pandas as pd
import seaborn as sns
from sklearn.utils import resample

df = pd.read_csv('data/interim/balanced.csv')

#%% Display top x rows
print(df.head(20))

#%% Display category distribution
sns.countplot(x='category', data=df)

#%% Display word count distribution
df['word_count'] = df['name'].str.split().apply(len)
sns.distplot(df['word_count'], kde=False,
    bins=range(1, 26))

#%%

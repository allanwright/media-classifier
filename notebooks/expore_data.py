#%% Import modules and read dataset
import pandas as pd
import seaborn as sns

df = pd.read_csv('data/interim/combined.csv')

#%% Display top x rows
print(df.head(20))

#%% Display category distribution
sns.countplot(x='category', data=df)
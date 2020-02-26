import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from mccore import preprocessing

# %%
df = pd.read_csv('../data/interim/combined.csv')
# Remove file sizes from the end of filenames
df['name'] = df['name'].str.replace(r'\s{1}\(.+\)$', '')
df['name'] = df['name'].str.replace(r' - \S+\s{1}\S+$', '')
# Create file extension column
ext = df['name'].str.extract(r'\.(\w{3})$')
ext.columns = ['ext']
df['ext'] = ext['ext']
# Remove paths from filenames
df['name'] = df['name'].str.split('/').str[-1]
# Process filenames
df['name'] = df['name'].apply(preprocessing.prepare_input)

# %%
resolutions = [ '576p', '720p', '1080p', '2160p', '4k' ]
classes = [ 'movie', 'tv' ]

# %%
df['res'] = 'none'
for res in resolutions:
    regex = r'\b%s\b' % res
    df.loc[df['category'].isin(classes) & df['name'].str.contains(regex), ['res']] = res

# %%
for c in classes:
    regex = r'^%s$' % c
    d = df[df['category'].str.match(regex)]['res'].value_counts()
    sns.barplot(d.index, d.values)
    plt.title(f'{c} by resolution')
    plt.show()
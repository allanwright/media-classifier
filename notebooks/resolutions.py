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
resolutions = [ '480p', '576p', '720p', '1080p', '2160p', '4k' ]
classes = [ 'movie', 'tv' ]

def set_res(df):
    df['res'] = 'none'
    for res in resolutions:
        regex = r'\b%s\b' % res
        df.loc[df['category'].isin(classes) & df['name'].str.contains(regex), ['res']] = res

def plot_res(df):
    for c in classes:
        regex = r'^%s$' % c
        d = df[df['category'].str.match(regex)]['res'].value_counts()
        print(d)
        sns.barplot(d.index, d.values)
        plt.title(f'{c} by resolution')
        plt.show()

# %%
set_res(df)
plot_res(df)

# %%
# Find out how many filenames contains both 4k and 2160p
r2160p = r'\b2160p\b'
r4k = r'\b4\b'
combined = df[df['name'].str.contains(r2160p) & df['name'].str.contains(r4k)]['category'].value_counts()
print(combined)

# %%
def process_input(name, res, other_res):
    words = name.split(' ')
    words = [res if i == other_res else i for i in words]
    return ' '.join(words)

df_res = pd.DataFrame()

for res in resolutions:
    if res == 'none':
        continue
    for other_res in resolutions:
        if other_res == 'none' or res == other_res:
            continue
        df_other = df[df['res'].str.match(r'^%s$' % other_res)].copy()
        df_other = df_other.copy()
        df_other['name'] = df['name'].apply(process_input, args=(res, other_res))
        df_res = df_res.append(df_other)

df = df.append(df_res)

# %%
set_res(df)
plot_res(df)
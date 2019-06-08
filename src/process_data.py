import os
import pandas as pd

def get_consolidated_raw_data(path):
    '''Gets a pandas dataframe containing the contents of all raw data files.

    The name of the folder used to store each file is used for the category
    column in the resulting dataframe.

    Args:
        path (string): The path of the raw data.
    
    Returns:
        DataFrame: The contents of all raw data files.
    '''
    consolidated = pd.DataFrame()
    for x in os.listdir(path):
        x_path = '%s/%s' % (path, x)
        if os.path.isdir(x_path):
            for y in os.listdir(x_path):
                y_path = '%s/%s/%s' % (path, x, y)
                if os.path.isfile(y_path):
                    print('Processing %s' % y_path)
                    series = pd.read_csv(y_path, sep='\t', squeeze=True)
                    df = pd.DataFrame()
                    df['name'] = series
                    df['category'] = x
                    consolidated = consolidated.append(df, ignore_index=True)
    return consolidated

def process_data():
    '''Processes the raw data files.
    '''
    df = get_consolidated_raw_data('../data/raw')

    # Remove commas from the name column
    df['name'] = df['name'].str.replace(',', '')

    # Remove file sizes from the end of filenames
    df['name'] = df['name'].str.replace(r'\s{1}\(.+\)$', '')

    #Merge categories
    df.loc[df['category'] == 'games', 'category'] = 'apps'

    # Create file extension column
    ext = df['name'].str.extract(r'\.(\w{3})$')
    ext.columns = ['ext']
    df['ext'] = ext['ext']

    # Remove file extension from filenames
    df['name'] = df['name'].str.replace(r'\.(\w{3})$', '')
    
    # Remove junk filenames
    music_ext = [
        'mp3',
        'm4a',
        'ogg',
        'flac',
        'wav'
    ]

    movie_ext = [
        'mp4',
        'mkv',
        'avi',
        'wmv',
        'mpg',
        'm4v'
    ]

    tv_ext = [
        'mp4',
        'mkv',
        'avi',
        'wmv',
        'mpg',
        'm4v'
    ]

    app_ext = [
        'exe',
        'bin',
        'zip',
        'rar',
        'iso',
        'cab',
        'dll',
        'msi',
        'dmg',
        'dat'
    ]

    df = df[((df['category'] == 'music') & (df['ext'].isin(music_ext))) |
            ((df['category'] == 'movies') & (df['ext'].isin(movie_ext))) |
            ((df['category'] == 'tv') & (df['ext'].isin(tv_ext))) |
            ((df['category'] == 'apps') & (df['ext'].isin(app_ext)))]
    
    # Remove duplicates by filename and category
    df.drop_duplicates(subset=['name', 'category'], inplace=True)

    # Normalize word separators
    df['name'] = df['name'].str.replace('.', ' ')
    df['name'] = df['name'].str.replace('_', ' ')
    df['name'] = df['name'].str.replace('-', ' ')
    df['name'] = df['name'].str.replace('[', ' ')
    df['name'] = df['name'].str.replace(']', ' ')
    df['name'] = df['name'].str.replace('+', ' ')
    df['name'] = df['name'].str.split().str.join(' ')

    df.to_csv('../data/interim/combined.csv', index=False)

process_data()
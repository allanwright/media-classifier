import json
import os
import pandas as pd
import progressbar as pb
import re
from sklearn import model_selection
from sklearn.utils import resample

def clean_workspace():
    '''Removes all processed data and models from the workspace.

    '''
    paths = [
        'data/interim',
        'data/processed',
        'models/baseline',
        'models/cnn'
    ]

    for path in paths:
        delete_files_from_dir(path)

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
    paths = ['movie', 'tv']
    for x in os.listdir(path):
        x_path = '%s/%s' % (path, x)
        if os.path.isdir(x_path) and x in paths:
            print('Consolidating {path}'.format(path=x_path))
            for y in pb.progressbar(os.listdir(x_path)):
                y_path = '%s/%s/%s' % (path, x, y)
                if os.path.isfile(y_path):
                    series = pd.read_csv(y_path, sep='\t', squeeze=True)
                    df = pd.DataFrame(data={'name': series, 'category': x})
                    consolidated = consolidated.append(df, ignore_index=True)
    return consolidated

def process_data():
    '''Processes the raw data files.
    '''
    df = get_consolidated_raw_data('data/raw')

    printProgress('Processing data', df)

    # Remove commas from the name column
    df['name'] = df['name'].str.replace(',', '')

    # Lowercase all names
    df['name'] = df['name'].str.lower()

    # Remove file sizes from the end of filenames
    df['name'] = df['name'].str.replace(r'\s{1}\(.+\)$', '')
    df['name'] = df['name'].str.replace(r' - \S+\s{1}\S+$', '')

    #Merge categories
    df.loc[df['category'] == 'game', 'category'] = 'app'

    # Create file extension column
    ext = df['name'].str.extract(r'\.(\w{3})$')
    ext.columns = ['ext']
    df['ext'] = ext['ext']

    # Remove file extension from filenames
    df['name'] = df['name'].str.replace(r'\.(\w{3})$', '')
    
    # Remove paths from filenames
    df['name'] = df['name'].str.split('/').str[-1]
    
    # Remove junk filenames
    movie_ext = [ 'mp4', 'mkv', 'avi', 'wmv', 'mpg', 'm4v' ]
    tv_ext = [ 'mp4', 'mkv', 'avi', 'wmv', 'mpg', 'm4v' ]

    df = df[((df['category'] == 'movie') & (df['ext'].isin(movie_ext))) |
            ((df['category'] == 'tv') & (df['ext'].isin(tv_ext)))]
    
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

    # Remove rubbish characters
    for c in '`~!@#$%^&*()-_+=[]|;:<>,./?\'':
        df['name'] = df['name'].str.replace(c, '')

    # Append extension to name column then drop extension column
    df['name'] = df['name'].map(str) + ' ' + df['ext']
    df = df.drop('ext', axis=1)

    # Split combined season and episode numbers
    df['name'] = df['name'].apply(split_season_episode)

    # Save interim output before processing further
    df.to_csv('data/interim/combined.csv', index=False)

    # Split the filename into individual words then stack the DataFrame
    df = pd.DataFrame(df['name'].str.split().tolist(), index=[df.index, df.category]).stack()
    df = df.reset_index()
    df.columns = ['index', 'category', 'pos', 'word']

    # Add entity column
    df['entity'] = ''

    # Label file extension
    df.loc[(df['word'].isin(movie_ext)) & (df.category == 'movie'), 'entity'] = 'ext'
    df.loc[(df['word'].isin(tv_ext)) & (df.category == 'tv'), 'entity'] = 'ext'

    # Label resolution
    resolutions = ['576p', '720p', '1080p', '2160p', '4k']
    df.loc[df['word'].isin(resolutions), 'entity'] = 'res'

    # Label encoding
    encodings = ['h264', 'h265', 'x264', 'x265']
    df.loc[df['word'].isin(encodings), 'entity'] = 'enc'

    # Label season number
    df.loc[df['word'].str.match(r'^s\d+$'), 'entity'] = 'sid'

    # Label episode number
    df.loc[df['word'].str.match(r'^e\d+$'), 'entity'] = 'eid'

    # Save interim stacked output before processing further
    df.to_csv('data/interim/stacked.csv', index=False)

    #df.loc[df.word.str.contains('^s\d+e\d+$'), 'entity'] = 'season_episode'

    """ # Downsample to fix class imbalance
    printProgress('Balancing classes', df)
    categories = [df[df.category == c] for c in df.category.unique()]
    sample_size = min([len(c) for c in categories])
    downsampled = [resample(c,
                            replace=False,
                            n_samples=sample_size,
                            random_state=123) for c in categories]
    df = pd.concat(downsampled)

    # Save final output before splitting
    df.to_csv('data/interim/balanced.csv', index=False) """

    """ # Perform train test data split
    printProgress('Splitting data', df)
    train, test = model_selection.train_test_split(df, test_size=0.2, random_state=123)
    x_train = train.drop('category', axis=1)
    y_train = train['category']
    x_test = test.drop('category', axis=1)
    y_test = test['category']

    # Save train and test data
    printProgress('Saving data', df)
    x_train.to_csv('data/processed/x_train.csv', index=False, header=False)
    y_train.to_csv('data/processed/y_train.csv', index=False, header=False)
    x_test.to_csv('data/processed/x_test.csv', index=False, header=False)
    y_test.to_csv('data/processed/y_test.csv', index=False, header=False) """

# TODO: Call this method from process_data using pandas apply()
def process_filename(filename):
    '''Processes a filename in preparation for classification by a model.
    '''
    # Remove commas
    filename = filename.replace(',', '')

    # Lowercase filename
    filename = filename.lower()

    # Remove file sizes
    filename = filename.replace(r'\s{1}\(.+\)$', '')
    filename = filename.replace(r' - \S+\s{1}\S+$', '')

    # Remove file extension
    filename = filename.replace(r'\.(\w{3})$', '')
    
    # Remove paths
    filename = filename.split('/')[-1]

    # Normalize word separators
    filename = filename.replace('.', ' ')
    filename = filename.replace('_', ' ')
    filename = filename.replace('-', ' ')
    filename = filename.replace('[', ' ')
    filename = filename.replace(']', ' ')
    filename = filename.replace('+', ' ')
    filename = ' '.join(filename.split())

    # Remove rubbish characters
    filename = filename.strip('`~!@#$%^&*()-_+=[]|;:<>,./?')

    # Split season and episode numbers
    filename = split_season_episode(filename)

    return filename

# TODO: Rename with underscores
def dictToJson(dict, path):
    '''Serializes a dictionary as json and writes it to the specified file path.

    Args:
        dict (dict): The dictionary to serialize.
        path (string): The file path to write to.
    '''
    dict_json = json.dumps(dict)
    with open(path, 'w') as json_file:
        json_file.write(dict_json)

def delete_files_from_dir(path):
    exclusions = ['.gitignore', '.gitkeep']
    print('Cleaning {path}'.format(path=path))
    for i in pb.progressbar(os.listdir(path)):
        file_path = path + '/' + i
        if os.path.isfile(file_path) and i not in exclusions:
            os.remove(file_path)

def printProgress(message, df):
    print('{message} ({rows} rows)'.format(message=message, rows=df.shape[0]))

def split_season_episode(name):
    patterns = [
        [r'(?P<sid>s\d+)(?P<eid>e\d+)', '{sid} {eid}'], #s01e01
        [r'(?P<sid>\d+)x(?P<eid>\d+)', 's{sid} e{eid}'] #01x01
    ]
    for pattern in patterns:
        match = re.search(pattern[0], name)
        if match != None:
            name = name.replace(
                match.group(0),
                pattern[1].format(
                    sid=match.group('sid'),
                    eid=match.group('eid')))
    return name
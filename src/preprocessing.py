import json
import os
import pandas as pd
import progressbar as pb
from sklearn import model_selection
from sklearn.utils import resample
from sklearn.preprocessing import LabelEncoder

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

def delete_files_from_dir(path):
    print('Cleaning {path}'.format(path=path))
    for i in pb.progressbar(os.listdir(path)):
        file_path = path + '/' + i
        if os.path.isfile(file_path):
            os.remove(file_path)

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
    music_ext = [ 'mp3', 'm4a', 'ogg', 'flac', 'wav' ]
    movie_ext = [ 'mp4', 'mkv', 'avi', 'wmv', 'mpg', 'm4v' ]
    tv_ext = [ 'mp4', 'mkv', 'avi', 'wmv', 'mpg', 'm4v' ]
    app_ext = [ 'exe', 'bin', 'zip', 'rar', 'iso',
                'cab', 'dll', 'msi', 'dmg', 'dat' ]

    df = df[((df['category'] == 'music') & (df['ext'].isin(music_ext))) |
            ((df['category'] == 'movie') & (df['ext'].isin(movie_ext))) |
            ((df['category'] == 'tv') & (df['ext'].isin(tv_ext))) |
            ((df['category'] == 'app') & (df['ext'].isin(app_ext)))]
    
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

    # Save interim output before processing further
    df.to_csv('data/interim/combined.csv', index=False)

    # Downsample to fix class imbalance
    printProgress('Balancing classes', df)
    categories = [df[df.category == c] for c in df.category.unique()]
    sample_size = min([len(c) for c in categories])
    downsampled = [resample(c,
                            replace=False,
                            n_samples=sample_size,
                            random_state=123) for c in categories]
    df = pd.concat(downsampled)

    # Encode labels
    labelEncoder = LabelEncoder()
    labelEncoder.fit(df['category'])
    df['category'] = labelEncoder.transform(df['category'])

    # Save label encoding
    category_ids = df.category.unique()
    category_names = labelEncoder.inverse_transform(category_ids)
    category_dict = {}
    for i in range(len(category_ids)):
        category_dict[int(category_ids[i])] = category_names[i]
    dictToJson(category_dict, 'data/processed/label_dictionary.json')

    # Save final output before splitting
    df.to_csv('data/interim/final.csv', index=False)

    # Perform train test data split
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
    y_test.to_csv('data/processed/y_test.csv', index=False, header=False)

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

    return filename

def dictToJson(dict, path):
    '''Serializes a dictionary as json and writes it to the specified file path.

    Args:
        dict (dict): The dictionary to serialize.
        path (string): The file path to write to.
    '''
    dict_json = json.dumps(dict)
    with open(path, 'w') as json_file:
        json_file.write(dict_json)

def printProgress(message, df):
    print('{message} ({rows} rows)'.format(message=message, rows=df.shape[0]))
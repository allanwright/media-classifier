import os
import pandas as pd
import progressbar as pb
import re
from sklearn import model_selection
from sklearn.utils import resample
from sklearn.preprocessing import LabelEncoder
from mccore import persistence
from mccore import preprocessing

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
                    df = pd.DataFrame(data={
                        'name': series,
                        'category': x})
                    consolidated = consolidated.append(df, ignore_index=True)
    return consolidated

# TODO: Reassign any filenames with season and episode number to tv category
def process_data():
    '''Processes the raw data files.
    '''
    df = get_consolidated_raw_data('data/raw')

    print_progress('Processing data for classification', df)

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

    #Merge categories
    df.loc[df['category'] == 'game', 'category'] = 'app'
    
    # Remove junk filenames
    music_ext = get_music_ext()
    movie_ext = get_movie_ext()
    tv_ext = get_tv_ext()
    app_ext = get_app_ext()

    df = df[((df['category'] == 'music') & (df['ext'].isin(music_ext))) |
            ((df['category'] == 'movie') & (df['ext'].isin(movie_ext))) |
            ((df['category'] == 'tv') & (df['ext'].isin(tv_ext))) |
            ((df['category'] == 'app') & (df['ext'].isin(app_ext)))]
    
    # Remove duplicates by filename and category
    df.drop_duplicates(subset=['name', 'category'], inplace=True)

    # Save interim output before processing further
    df.to_csv('data/interim/combined.csv', index=False)

    # Downsample to fix class imbalance
    print_progress('Balancing classes', df)
    categories = [df[df.category == c] for c in df.category.unique()]
    sample_size = min([len(c) for c in categories])
    downsampled = [resample(c,
                            replace=False,
                            n_samples=sample_size,
                            random_state=123) for c in categories]
    df = pd.concat(downsampled)

    # Save final output before splitting
    df.to_csv('data/interim/balanced.csv', index=False)

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
    persistence.obj_to_json(
        category_dict, 'data/processed/label_dictionary.json')
    
    # Create named entity columns
    df['title'] = ''
    df['source'] = ''
    df['season_id'] = ''
    df['episode_id'] = ''
    df['episode_name'] = ''
    df['resolution'] = ''
    df['encoding'] = ''
    df['year'] = ''
    df['extension'] = ''

    # Save final output before splitting
    df.to_csv('data/interim/final.csv', index=False)

    # Perform train test data split
    print_progress('Splitting data', df)
    train, test = model_selection.train_test_split(df, test_size=0.2, random_state=123)
    x_train = train.drop('category', axis=1)
    y_train = train['category']
    x_test = test.drop('category', axis=1)
    y_test = test['category']

    # Save classification train and validation data
    print_progress('Saving data', df)
    x_train.to_csv('data/processed/x_train.csv', index=False, header=False)
    y_train.to_csv('data/processed/y_train.csv', index=False, header=False)
    x_test.to_csv('data/processed/x_test.csv', index=False, header=False)
    y_test.to_csv('data/processed/y_test.csv', index=False, header=False)

    # Process data for named entity recognition
    #process_data_for_ner()

def apply_entity_names(row, nlp):
    doc = nlp(row['name'])
    for ent in doc.ents:
        print(ent)

# TODO: Remove filenames containing 2MovieRulz
# TODO: Remove filenames containing desi
def process_data_for_ner():
    df = pd.read_csv('data/interim/combined.csv')
    print_progress('Processing data for named entity recognition', df)

    # Remove all rows that aren't a movie or tv show
    df = df[df['category'].isin(['movie', 'tv'])]

    # Remove as many foreign language filenames as possible
    df = df[~df.name.str.contains('tamil')]
    df = df[~df.name.str.contains('hindi')]
    df = df[~df.name.str.contains('www')]

    df.to_csv('data/interim/pruned.csv', index=False)

    # Downsample to fix class imbalance
    print_progress('Balancing classes', df)
    categories = [df[df.category == c] for c in df.category.unique()]
    sample_size = 100 #min([len(c) for c in categories])
    downsampled = [resample(c,
                            replace=False,
                            n_samples=sample_size,
                            random_state=123) for c in categories]
    df = pd.concat(downsampled)

    # Save final output before splitting
    df.to_csv('data/interim/pruned_balanced.csv', index=False)

    # Split the filename into individual words then stack the DataFrame
    print_progress('Stacking dataset', df)
    df = pd.DataFrame(df['name'].str.split().tolist(), index=[df.index, df.category]).stack()
    df = df.reset_index()
    df.columns = ['index', 'category', 'pos', 'word']

    # Add entity column
    df['entity'] = ''

    # Label file extension
    movie_ext = get_movie_ext()
    tv_ext = get_tv_ext()
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

    # Label source
    sources = ['bluray', 'brrip', 'web', 'webrip', 'hdtv', 'hdrip',
               'dvd', 'dvdrip']
    df.loc[df['word'].isin(sources), 'entity'] = 'src'

    # Label year
    years = [str(x) for x in range(1940, 2020)]
    df.loc[df['word'].isin(years), 'entity'] = 'year'

    # Save interim stacked output before processing further
    df.to_csv('data/interim/stacked.csv', index=False)

def get_app_ext():
    return [ 'exe', 'bin', 'zip', 'rar', 'iso',
             'cab', 'dll', 'msi', 'dmg', 'dat' ]

def get_movie_ext():
    return [ 'mp4', 'mkv', 'avi', 'wmv', 'mpg', 'm4v' ]

def get_music_ext():
    return [ 'mp3', 'm4a', 'ogg', 'flac', 'wav' ]

def get_tv_ext():
    return [ 'mp4', 'mkv', 'avi', 'wmv', 'mpg', 'm4v' ]

def print_progress(message, df):
    print('{message} ({rows} rows)'.format(message=message, rows=df.shape[0]))
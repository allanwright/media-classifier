'''Aquisition of data for training.

'''

import json
import pickle

from mccore import persistence
from mccore import preprocessing
from sklearn import model_selection
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import resample
import pandas as pd

from src.pipelines.process import Process

def process_all():
    ''' Performs all data processing steps.

    '''
    process_merge()
    process_feature()

def process_merge():
    ''' Performs merging of data.

    '''
    pipeline = Process()
    pipeline.run()

def process_feature():
    ''' Performs feature generation.

    '''
    df = pd.read_csv('data/interim/combined.csv')

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

    # Augment training data with all resolutions
    print_progress('Augmenting training data', df)
    df['res'] = ''
    resolutions = get_resolutions()
    df_res = pd.DataFrame()

    for res in resolutions:
        if res == 'none':
            continue
        for other_res in resolutions:
            if other_res == 'none' or res == other_res:
                continue
            df_other = df[df['res'].str.match(r'^%s$' % other_res)].copy()
            df_other = df_other.copy()
            df_other['name'] = df['name'].apply(find_and_replace, args=(res, other_res))
            pd.concat([df_res, df_other])

    #df = df.append(df_res)
    pd.concat([df, df_res])
    df = df.drop('res', axis=1)

    df.to_csv('data/interim/augmented.csv')

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
    df.to_csv('data/interim/cleaned.csv', index=False)

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
    label_encoder = LabelEncoder()
    label_encoder.fit(df['category'])
    df['category'] = label_encoder.transform(df['category'])

    # Save label encoding
    category_ids = df.category.unique()
    category_names = label_encoder.inverse_transform(category_ids)
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
    x_eval = test.drop('category', axis=1)
    y_eval = test['category']

    # Save classification train and validation data
    print_progress('Saving data', df)
    x_train.to_csv('data/processed/x_train.csv', index=False, header=False)
    y_train.to_csv('data/processed/y_train.csv', index=False, header=False)
    x_eval.to_csv('data/processed/x_eval.csv', index=False, header=False)
    y_eval.to_csv('data/processed/y_eval.csv', index=False, header=False)

    # Process test data
    df = pd.read_csv('data/raw/predictions/predictions.csv')
    df['name'] = df['name'].apply(preprocessing.prepare_input)
    x_test = df['name']
    y_test = df['class']
    x_test.to_csv('data/processed/x_test.csv', index=False, header=False)
    y_test.to_csv('data/processed/y_test.csv', index=False, header=False)

    # Process data for named entity recognition labelling
    process_data_for_ner()

    # Process labelled named entity recognition data (if any)
    process_labelled_ner_data()

def find_and_replace(sentence, find, replace):
    '''Finds a word in a sentence and replaces it with another word.

    Args:
        sentence (string): The sentence to perform the find and replace.
        find (string): The word to find.
        replace (string): The word to use for replacement.

    Returns:
        string: The sentence.
    '''
    words = sentence.split(' ')
    words = [find if i == replace else i for i in words]
    return ' '.join(words)

def apply_entity_names(row, nlp):
    doc = nlp(row['name'])
    for ent in doc.ents:
        print(ent)

def process_data_for_ner():
    '''Processes data for named entity recognition.

    '''
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
    sample_size = min([len(c) for c in categories])
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
    resolutions = get_resolutions()
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

def process_labelled_ner_data():
    '''Processes labelled data for named entity recongition.

    '''
    df = pd.read_csv('data/interim/ner_labelled.csv')

    # Keep only word and corresponding label
    df = df[['word', 'entity']]

    # Save to tsv
    df.to_csv(
        'data/interim/ner_labelled.tsv',
        sep='\t',
        header=False,
        index=False)

    # Convert from tsv to json
    tsv_to_json_format(
        "data/interim/ner_labelled.tsv",
        'data/interim/ner_labelled.json',
        'na')

    # Write out spacy file
    write_spacy_file(
        'data/interim/ner_labelled.json',
        'data/processed/ner_labelled.pickle')

def get_app_ext():
    '''Get app file extensions.

    '''
    return ['exe', 'bin', 'zip', 'rar', 'iso', 'cab', 'dll', 'msi', 'dmg', 'dat']

def get_movie_ext():
    '''Get movie file extensions.

    '''
    return ['mp4', 'mkv', 'avi', 'wmv', 'mpg', 'm4v']

def get_music_ext():
    '''Get music file extensions.

    '''
    return ['mp3', 'm4a', 'ogg', 'flac', 'wav']

def get_tv_ext():
    '''Get tv file extensions.

    '''
    return ['mp4', 'mkv', 'avi', 'wmv', 'mpg', 'm4v']

def get_resolutions():
    '''Get resolutions.

    '''
    return ['480p', '576p', '720p', '1080p', '2160p', '4k']

def print_progress(message, df):
    '''Prints a message about the progress of the data processing.

    Args:
        message (string): The message to print.
        df (DataFrame): The current DataFrame.
    '''
    print('{message} ({rows} rows)'.format(message=message, rows=df.shape[0]))

def tsv_to_json_format(input_path, output_path, unknown_label):
    '''Converts TSV data to JSON.

    Args:
        input_path (string): The input path.
        output_path (string): the output path.
        unknown_label (string): The label to use for unknown entities.
    '''
    try:
        input_file = open(input_path, 'r') # input file
        output_file = open(output_path, 'w') # output file
        data_dict = {}
        annotations = []
        label_dict = {}
        words = ''
        start = 0
        for line in input_file:
            word, entity = line.split('\t')
            words += word + " "
            entity = entity[:len(entity)-1]
            if entity != unknown_label:
                if len(entity) != 1:
                    d = {}
                    d['text'] = word
                    d['start'] = start
                    d['end'] = start+len(word) - 1
                    try:
                        label_dict[entity].append(d)
                    except:
                        label_dict[entity] = []
                        label_dict[entity].append(d)
            start += len(word) + 1
            if entity == 'extension':
                data_dict['content'] = words
                words = ''
                label_list = []
                for ents in list(label_dict.keys()):
                    for i in range(len(label_dict[ents])):
                        if label_dict[ents][i]['text'] != '':
                            l = [ents, label_dict[ents][i]]
                            for j in range(i + 1, len(label_dict[ents])):
                                if label_dict[ents][i]['text'] == label_dict[ents][j]['text']:
                                    di = {}
                                    di['start'] = label_dict[ents][j]['start']
                                    di['end'] = label_dict[ents][j]['end']
                                    di['text'] = label_dict[ents][i]['text']
                                    l.append(di)
                                    label_dict[ents][j]['text'] = ''
                            label_list.append(l)

                for entities in label_list:
                    label = {}
                    label['label'] = [entities[0]]
                    label['points'] = entities[1:]
                    annotations.append(label)
                data_dict['annotation'] = annotations
                annotations = []
                json.dump(data_dict, output_file)
                output_file.write('\n')
                data_dict = {}
                start = 0
                label_dict = {}
    except Exception as e:
        print("Unable to process file" + "\n" + "error = " + str(e))
        return None

def write_spacy_file(input_file=None, output_file=None):
    '''Writes the specified input to a spacy file.

    Args:
        input_file (file): The input file.
        output_file (file): the output file.
    '''
    try:
        training_data = []
        lines = []
        with open(input_file, 'r') as f:
            lines = f.readlines()

        for line in lines:
            data = json.loads(line)
            text = data['content']
            entities = []
            for annotation in data['annotation']:
                point = annotation['points'][0]
                labels = annotation['label']
                if not isinstance(labels, list):
                    labels = [labels]

                for label in labels:
                    entities.append((point['start'], point['end'] + 1, label))

            training_data.append((text, {"entities" : entities}))

        with open(output_file, 'wb') as fp:
            pickle.dump(training_data, fp)

    except Exception as e:
        print("Unable to process " + input_file + "\n" + "error = " + str(e))
        return None

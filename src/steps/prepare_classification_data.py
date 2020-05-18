'''Defines a pipeline step which prepares training and test data for
media classification.

'''

from mccore import persistence
from mccore import preprocessing
from sklearn import model_selection
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import resample
import pandas as pd

from src.step import Step
import src.preprocessing as src_preprocessing

class PrepareClassificationData(Step):
    '''Defines a pipeline step which prepares training and test data for
    media classification.

    '''

    def __init__(self):
        super(PrepareClassificationData, self).__init__()
        self.input = {
            'combined': 'data/interim/combined.csv',
            'predictions': 'data/predictions/predictions.csv',
        }
        self.output = {
            'augmented': 'data/interim/augmented.csv',
            'cleaned': 'data/interim/cleaned.csv',
            'balanced': 'data/interim/balanced.csv',
            'final': 'data/interim/final.csv',
            'x_train': 'data/processed/x_train.csv',
            'y_train': 'data/processed/y_train.csv',
            'x_eval': 'data/processed/x_eval.csv',
            'y_eval': 'data/processed/y_eval.csv',
            'x_test': 'data/processed/x_test.csv',
            'y_test': 'data/processed/y_test.csv',
        }

    def run(self):
        '''Runs the pipeline step.

        '''
        df = pd.read_csv(self.input['combined'])
        self.print('Processing data for classification ({rows} rows)', rows=df.shape[0])

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
        self.print('Augmenting training data ({rows} rows)', rows=df.shape[0])
        df['res'] = ''
        resolutions = src_preprocessing.get_resolutions()
        df_res = pd.DataFrame()

        for res in resolutions:
            if res == 'none':
                continue
            for other_res in resolutions:
                if other_res == 'none' or res == other_res:
                    continue
                df_other = df[df['res'].str.match(r'^%s$' % other_res)].copy()
                df_other = df_other.copy()
                df_other['name'] = df['name'].apply(self.__find_and_replace, args=(res, other_res))
                pd.concat([df_res, df_other])

        pd.concat([df, df_res])
        df = df.drop('res', axis=1)

        df.to_csv(self.output['augmented'])

        #Merge categories
        df.loc[df['category'] == 'game', 'category'] = 'app'

        # Remove junk filenames
        music_ext = src_preprocessing.get_music_ext()
        movie_ext = src_preprocessing.get_movie_ext()
        tv_ext = src_preprocessing.get_tv_ext()
        app_ext = src_preprocessing.get_app_ext()

        df = df[((df['category'] == 'music') & (df['ext'].isin(music_ext))) |
                ((df['category'] == 'movie') & (df['ext'].isin(movie_ext))) |
                ((df['category'] == 'tv') & (df['ext'].isin(tv_ext))) |
                ((df['category'] == 'app') & (df['ext'].isin(app_ext)))]

        # Remove duplicates by filename and category
        df.drop_duplicates(subset=['name', 'category'], inplace=True)

        # Save interim output before processing further
        df.to_csv(self.output['cleaned'], index=False)

        # Downsample to fix class imbalance
        self.print('Balancing classes ({rows} rows)', rows=df.shape[0])
        categories = [df[df.category == c] for c in df.category.unique()]
        sample_size = min([len(c) for c in categories])
        downsampled = [resample(c,
                                replace=False,
                                n_samples=sample_size,
                                random_state=123) for c in categories]
        df = pd.concat(downsampled)

        # Save final output before splitting
        df.to_csv(self.output['balanced'], index=False)

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

        # Save final output before splitting
        df.to_csv(self.output['final'], index=False)

        # Perform train test data split
        self.print('Splitting data ({rows} rows)', rows=df.shape[0])
        train, test = model_selection.train_test_split(df, test_size=0.2, random_state=123)
        x_train = train.drop('category', axis=1)
        y_train = train['category']
        x_eval = test.drop('category', axis=1)
        y_eval = test['category']

        # Save classification train and validation data
        self.print('Saving data ({rows} rows)', rows=df.shape[0])
        x_train.to_csv(self.output['x_train'], index=False, header=False)
        y_train.to_csv(self.output['y_train'], index=False, header=False)
        x_eval.to_csv(self.output['x_eval'], index=False, header=False)
        y_eval.to_csv(self.output['y_eval'], index=False, header=False)

        # Process test data
        df = pd.read_csv(self.input['predictions'])
        df['name'] = df['name'].apply(preprocessing.prepare_input)
        x_test = df['name']
        y_test = df['class']
        x_test.to_csv(self.output['x_test'], index=False, header=False)
        y_test.to_csv(self.output['y_test'], index=False, header=False)

    def __find_and_replace(self, sentence, find, replace):
        words = sentence.split(' ')
        words = [find if i == replace else i for i in words]
        return ' '.join(words)

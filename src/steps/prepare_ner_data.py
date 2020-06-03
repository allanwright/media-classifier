'''Defines a pipeline step which prepares training and test data for
named entity recognition.

'''

import json
import pickle

import pandas as pd

from src.step import Step
import src.preprocessing as preprocessing

class PrepareNerData(Step):
    '''Defines a pipeline step which prepares training and test data for
    named entity recognition.

    '''

    def __init__(self):
        super(PrepareNerData, self).__init__()
        self.input = {
            'processed': 'data/interim/processed.csv',
            'ner_labelled_csv': 'data/interim/ner_labelled.csv',
        }
        self.output = {
            'stacked': 'data/interim/stacked.csv',
            'ner_labelled_tsv': 'data/interim/ner_labelled.tsv',
            'ner_labelled_json': 'data/interim/ner_labelled.json',
            'ner_labelled_pickle': 'data/interim/ner_labelled.pickle',
        }

    def run(self):
        '''Runs the pipeline step.

        '''

        # Process data for named entity recognition labelling
        self.__process_data_for_ner()

        # Process labelled named entity recognition data (if any)
        self.__process_labelled_ner_data()

    def __apply_entity_names(self, row, nlp):
        doc = nlp(row['name'])
        for ent in doc.ents:
            print(ent)

    def __process_data_for_ner(self):
        df = pd.read_csv(self.input['processed'])
        self.print('Processing data for named entity recognition ({rows} rows)', rows=df.shape[0])

        # Split the filename into individual words then stack the DataFrame
        self.print('Stacking dataset ({rows} rows)', rows=df.shape[0])
        df = pd.DataFrame(df['name'].str.split().tolist(), index=[df.index, df.category]).stack()
        df = df.reset_index()
        df.columns = ['index', 'category', 'pos', 'word']

        # Add entity column
        df['entity'] = ''

        # Label file extension
        ext = preprocessing.get_app_ext()
        df.loc[df['word'].isin(app_ext) & df['category'].isin([0]), 'entity'] = 'ext'

        music_ext = preprocessing.get_music_ext()
        df.loc[df['word'].isin(music_ext) & df['category'].isin([2]), 'entity'] = 'ext'

        movie_tv_ext = preprocessing.get_movie_tv_ext()
        categories = [1, 3]
        df.loc[df['word'].isin(movie_tv_ext) & df['category'].isin(categories), 'entity'] = 'ext'

        # Label resolution
        resolutions = preprocessing.get_resolutions()
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
        df.to_csv(self.output['stacked'], index=False)

    def __process_labelled_ner_data(self):
        df = pd.read_csv(self.input['ner_labelled_csv'])

        # Keep only word and corresponding label
        df = df[['word', 'entity']]

        # Save to tsv
        df.to_csv(
            self.output['ner_labelled_tsv'],
            sep='\t',
            header=False,
            index=False)

        # Convert from tsv to json
        self.__tsv_to_json_format(
            self.output['ner_labelled_tsv'],
            self.output['ner_labelled_json'],
            'na')

        # Write out spacy file
        self.__write_spacy_file(
            self.output['ner_labelled_json'],
            self.output['ner_labelled_pickle'])

    def __tsv_to_json_format(self, input_path, output_path, unknown_label):
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

    def __write_spacy_file(self, input_file=None, output_file=None):
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

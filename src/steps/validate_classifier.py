'''Defines a pipeline step which validates the classification model.

'''

import pandas as pd

from mccore import persistence
from mccore import Classifier

from src.step import Step

class ValidateClassifier(Step):
    '''Defines a pipeline step which validates the classification model.

    '''

    def __init__(self):
        super(ValidateClassifier, self).__init__()
        self.input = {
            'predictions': 'data/test/classifier.csv',
            'label_dict': 'data/processed/label_dictionary.json',
            'vectorizer': 'models/classifier_vec.pickle',
            'model': 'models/classifier_mdl.pickle',
        }

    def run(self):
        '''Runs the pipeline step.

        '''
        labels = persistence.json_to_obj(self.input['label_dict'])
        classifier = Classifier(
            persistence.bin_to_obj(self.input['TrainClassifier_vectorizer']),
            persistence.bin_to_obj(self.input['TrainClassifier_model']),
            labels
        )

        def update_label(row):
            return labels[str(row['expected'])]

        def predict(row):
            classification = classifier.predict(row['name'])
            return classification['label']['name']

        df = pd.read_csv(self.input['predictions'])
        df['expected'] = df.apply(update_label, axis=1)
        df['actual'] = df.apply(predict, axis=1)

        def print_incorrect(row):
            if row['actual'] != row['expected']:
                self.print(
                    '\'{name}\' [ expected: {expected}, actual: {actual} ]',
                    name=row['name'],
                    expected=row['expected'],
                    actual=row['actual'])

        df.apply(print_incorrect, axis=1)

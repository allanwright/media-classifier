'''Defines a pipeline step which validates the classification model.

'''

import pandas as pd

from mccore import persistence
from mccore import Classifier

from src.step import Step

class ValidateClassificationModel(Step):
    '''Defines a pipeline step which validates the classification model.

    '''

    def __init__(self):
        super(ValidateClassificationModel, self).__init__()
        self.input = {
            'predictions': 'data/predictions/predictions.csv',
        }
        self.output = {
        }

    def run(self):
        '''Runs the pipeline step.

        '''
        labels = persistence.json_to_obj('data/processed/label_dictionary.json')
        classifier = Classifier(
            persistence.bin_to_obj('models/cls_base_vec.pickle'),
            persistence.bin_to_obj('models/cls_base_mdl.pickle'),
            labels
        )

        def update_label(row):
            return labels[str(row['expected'])]

        def predict(row):
            label, _ = classifier.predict(row['name'])
            return label['name']

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

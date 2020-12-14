'''Defines a pipeline step which validates the ner model.

'''

import ast

import pandas as pd

from mccore import EntityRecognizer
from mccore import ner
from mccore import persistence

from src.step import Step

class ValidateNer(Step):
    '''Defines a pipeline step which validates the ner model.

    '''

    def __init__(self):
        super(ValidateNer, self).__init__()
        self.input = {
            'predictions': 'data/test/ner.csv',
            'model': 'models/ner_mdl.pickle',
        }

    def run(self):
        '''Runs the pipeline step.

        '''
        nlp, _ = ner.get_model()
        nlp.from_bytes(persistence.bin_to_obj(self.input['model']))
        recognizer = EntityRecognizer(nlp)

        def predict(row):
            return recognizer.predict(row['name'])

        df = pd.read_csv(self.input['predictions'])
        df['actual'] = df.apply(predict, axis=1)

        def print_incorrect(row):
            actual_list = list(row['actual'])
            expected_list = list(ast.literal_eval(row['expected']))

            if len(actual_list) != len(expected_list):
                self.print(
                    '\'{name}\' [ expected: {expected}, actual: {actual} ]',
                    name=row['name'],
                    expected=row['expected'],
                    actual=row['actual'])
            else:
                for i in range(len(actual_list)): # pylint: disable=consider-using-enumerate
                    x = actual_list[i]
                    y = expected_list[i]
                    if x[0] != y[0] or x[1] != y[1]:
                        self.print(
                            '\'{name}\' [ expected: {expected}, actual: {actual} ]',
                            name=row['name'],
                            expected=y,
                            actual=x)

        df.apply(print_incorrect, axis=1)

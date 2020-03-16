'''Defines a pipeline step which merges training data.

'''

import os

import pandas as pd
import progressbar as pb

from src.step import Step

class Merge(Step):
    '''Defines a pipeline step which merges training data.

    '''

    def __init__(self):
        '''Initializes a new instance of the Merge object.

        '''
        self.input = {
            'path': 'data/raw',
            'excludes': ['predictions']
        }
        self.output = {
            'path': 'data/interim/combined.csv',
        }

    def run(self):
        '''Runs the pipeline step.

        '''
        path = self.input['path']
        excludes = self.input['excludes']
        consolidated = pd.DataFrame()
        for x in os.listdir(path):
            if x in excludes:
                continue
            x_path = '%s/%s' % (path, x)
            if os.path.isdir(x_path):
                print('Consolidating {path}'.format(path=x_path))
                for y in pb.progressbar(os.listdir(x_path)):
                    y_path = '%s/%s/%s' % (path, x, y)
                    if os.path.isfile(y_path):
                        series = pd.read_csv(y_path, sep='\t', squeeze=True)
                        df = pd.DataFrame(data={'name': series, 'category': x})
                        consolidated = consolidated.append(df, ignore_index=True)
        print_progress('Saving merged data', consolidated)
        consolidated.to_csv(self.output['path'], index=False)

def print_progress(message, df):
    '''Prints a message about the progress of the pipeline step.

    '''
    print('{message} ({rows} rows)'.format(message=message, rows=df.shape[0]))

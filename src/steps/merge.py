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
            'excludes': ['data/raw/predictions']
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
        for root, _, files in os.walk(path):
            if root == path or root in excludes:
                continue
            else:
                self.print('Consolidating {path}', path=root)
                for file in pb.progressbar(files):
                    series = pd.read_csv(os.path.join(root, file), sep='\t', squeeze=True)
                    df = pd.DataFrame(data={'name': series, 'category': root.split('/')[-1]})
                    consolidated = consolidated.append(df, ignore_index=True)
        self.print('Saving merged data ({rows} rows)', rows=consolidated.shape[0])
        consolidated.to_csv(self.output['path'], index=False)

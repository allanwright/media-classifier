'''Defines a pipeline step which aquires data from the kraken data source.

'''

import os

from src import datasets
from src.step import Step

class GetKrakenData(Step):
    '''Defines a pipeline step which aquires data from the kraken data source.

    '''

    def __init__(self):
        '''Initializes a new instance of the GetKrakenData object.

        '''
        super(GetKrakenData, self).__init__()
        self.input = {
            'path': os.getenv('KRAKEN_URL'),
        }
        self.output = {
            'movie': 'data/raw/movies/kraken.txt',
            'music': 'data/raw/music/kraken.txt',
            'tv': 'data/raw/tv/kraken.txt',
        }

    def run(self):
        '''Runs the pipeline step.

        '''
        for (key, value) in self.output.items():
            datasets.write_list_to_file(self.input['path'] % key, value)

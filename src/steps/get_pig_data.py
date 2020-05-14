'''Defines a pipeline step which aquires data from the pig data source.

'''

import os

from src import datasets
from src.step import Step

class GetPigData(Step):
    '''Defines a pipeline step which aquires data from the pig data source.

    '''

    def __init__(self):
        '''Initializes a new instance of the GetPigData object.

        '''
        super(GetPigData, self).__init__()
        self.input = {
            'path': os.getenv('PIG_PATH'),
        }
        self.output = {
            'Movies': 'data/raw/movies/pig.txt',
            'Music': 'data/raw/music/pig.txt',
            'TV Shows': 'data/raw/tv/pig.txt',
        }

    def run(self):
        '''Runs the pipeline step.

        '''
        for (key, value) in self.output.items():
            datasets.write_list_to_file(
                datasets.get_local_files(self.input['path'] % key), value)

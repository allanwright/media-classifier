'''Defines a processing pipeline that aquires test data.

'''

from src.pipeline import Pipeline
from src.steps.get_test_data import GetTestData

class AquireTestData(Pipeline):
    '''Defines a processing pipeline that aquires test data.

    '''

    def __init__(self):
        '''Initializes a new instance of the AquireTestData object.

        '''
        super(AquireTestData, self).__init__()
        self.add_step(GetTestData())

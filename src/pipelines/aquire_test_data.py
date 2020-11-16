'''Defines a processing pipeline that aquires test data.

'''

from src.pipeline import Pipeline
from src.steps.get_test_data import GetTestData

class AquireTestData(Pipeline):
    '''Defines a processing pipeline that aquires test data.

    '''

    def __init__(self, args):
        '''Initializes a new instance of the AquireTestData object.

        Args:
            args (dict): Dictionary of arguments that can be passed as input to every step.
        '''
        super(AquireTestData, self).__init__(args)
        self.add_step(GetTestData())

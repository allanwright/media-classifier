'''Defines a processing pipeline that prepares classification data for training.

'''

from src.pipeline import Pipeline
from src.steps.prepare_classification_data import PrepareClassificationData

class ProcessClassifier(Pipeline):
    '''Defines a processing pipeline that prepares classification data for training.

    '''

    def __init__(self, args):
        '''Initializes a new instance of the ProcessClassifier object.

        Args:
            args (dict): Dictionary of arguments that can be passed as input to every step.
        '''
        super(ProcessClassifier, self).__init__(args)
        self.add_step(PrepareClassificationData())

'''Defines a processing pipeline that prepares classification data for training.

'''

from src.pipeline import Pipeline
from src.steps.prepare_classification_data import PrepareClassificationData

class ProcessClassifier(Pipeline):
    '''Defines a processing pipeline that prepares classification data for training.

    '''

    def __init__(self):
        '''Initializes a new instance of the ProcessClassifier object.

        '''
        super(ProcessClassifier, self).__init__()
        self.add_step(PrepareClassificationData())

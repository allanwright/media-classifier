'''Defines a processing pipeline that prepares data for training.

'''

from src.pipeline import Pipeline
from src.steps.merge import Merge
from src.steps.prepare_classification_data import PrepareClassificationData
from src.steps.prepare_ner_data import PrepareNerData

class ProcessData(Pipeline):
    '''Defines a processing pipeline that prepares data for training.

    '''

    def __init__(self):
        '''Initializes a new instance of the ProcessData object.

        '''
        super(ProcessData, self).__init__()
        self.add_step(Merge())
        self.add_step(PrepareClassificationData())
        self.add_step(PrepareNerData())

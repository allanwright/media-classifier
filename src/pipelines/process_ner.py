'''Defines a processing pipeline that prepares ner data for training.

'''

from src.pipeline import Pipeline
from src.steps.prepare_ner_data import PrepareNerData

class ProcessNer(Pipeline):
    '''Defines a processing pipeline that prepares ner data for training.

    '''

    def __init__(self):
        '''Initializes a new instance of the ProcessNer object.

        '''
        super(ProcessNer, self).__init__()
        self.add_step(PrepareNerData())

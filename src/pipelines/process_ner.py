'''Defines a processing pipeline that prepares ner data for training.

'''

from src.pipeline import Pipeline
from src.steps.prepare_ner_data import PrepareNerData

class ProcessNer(Pipeline):
    '''Defines a processing pipeline that prepares ner data for training.

    '''

    def __init__(self, args):
        '''Initializes a new instance of the ProcessNer object.

        Args:
            args (dict): Dictionary of arguments that can be passed as input to every step.
        '''
        super(ProcessNer, self).__init__(args)
        self.add_step(PrepareNerData())

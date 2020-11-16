'''Defines a processing pipeline that trains the ner model.

'''

from src.pipeline import Pipeline
from src.steps.train_ner_model import TrainNerModel
from src.steps.validate_ner import ValidateNer

class TrainNer(Pipeline):
    '''Defines a processing pipeline that trains the ner model.

    '''

    def __init__(self, args):
        '''Initializes a new instance of the TrainNer object.

        Args:
            args (dict): Dictionary of arguments that can be passed as input to every step.
        '''
        super(TrainNer, self).__init__(args)
        self.add_step(TrainNerModel())
        self.add_step(ValidateNer())

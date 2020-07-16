'''Defines a processing pipeline that trains the ner model.

'''

from src.pipeline import Pipeline
from src.steps.train_ner_model import TrainNerModel
from src.steps.validate_ner import ValidateNer

class TrainNer(Pipeline):
    '''Defines a processing pipeline that trains the ner model.

    '''

    def __init__(self):
        super(TrainNer, self).__init__()
        self.add_step(TrainNerModel())
        self.add_step(ValidateNer())

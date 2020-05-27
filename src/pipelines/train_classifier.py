'''Defines a processing pipeline that trains the classification model.

'''

from src.pipeline import Pipeline
from src.steps.train_classifier import TrainClassifier as TrainClassifierStep
from src.steps.validate_classifier import ValidateClassifier

class TrainClassifier(Pipeline):
    '''Defines a processing pipeline that trains the classification model.

    '''

    def __init__(self):
        '''Initializes a new instance of the TrainClassifier object.

        '''
        super(TrainClassifier, self).__init__()
        self.add_step(TrainClassifierStep())
        self.add_step(ValidateClassifier())

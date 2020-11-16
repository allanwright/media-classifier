'''Defines a processing pipeline that trains the classification model.

'''

from src.pipeline import Pipeline
from src.steps.train_classifier import TrainClassifier as TrainClassifierStep
from src.steps.validate_classifier import ValidateClassifier

class TrainClassifier(Pipeline):
    '''Defines a processing pipeline that trains the classification model.

    '''

    def __init__(self, args):
        '''Initializes a new instance of the TrainClassifier object.

        Args:
            args (dict): Dictionary of arguments that can be passed as input to every step.
        '''
        super(TrainClassifier, self).__init__(args)
        self.add_step(TrainClassifierStep())
        self.add_step(ValidateClassifier())

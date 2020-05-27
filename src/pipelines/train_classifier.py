'''Defines a processing pipeline that trains the classification model.

'''

from src.pipeline import Pipeline
from src.steps.train_classification_model import TrainClassificationModel
from src.steps.validate_classification_model import ValidateClassificationModel

class TrainClassifier(Pipeline):
    '''Defines a processing pipeline that trains the classification model.

    '''

    def __init__(self):
        '''Initializes a new instance of the TrainClassifier object.

        '''
        super(TrainClassifier, self).__init__()
        self.add_step(TrainClassificationModel())
        self.add_step(ValidateClassificationModel())

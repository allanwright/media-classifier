'''Defines a processing pipeline that trains the classification model.

'''

from src.pipeline import Pipeline
from src.steps.promote_classifier import PromoteClassifier as PromoteClassifierStep

class PromoteClassifier(Pipeline):
    '''Defines a processing pipeline that promotes a classification model to production.

    '''

    def __init__(self):
        '''Initializes a new instance of the PromoteClassifier object.

        '''
        super(PromoteClassifier, self).__init__()
        self.add_step(PromoteClassifierStep())

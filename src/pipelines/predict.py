'''Defines a processing pipeline that uses a model to make a prediction.

'''

from src.pipeline import Pipeline
from src.steps.predict_classifier import PredictClassifier
from src.steps.predict_ner import PredictNer

class Predict(Pipeline):
    '''Defines a processing pipeline that uses a model to make a prediction.

    '''

    def __init__(self, args):
        '''Initializes a new instance of the Predict object.

        Args:
            args (dict): Dictionary of arguments that can be passed as input to every step.
        '''
        super(Predict, self).__init__(args)
        if args['--model'] == 'ner':
            self.add_step(PredictNer())
        elif args['--model'] == 'classifier':
            self.add_step(PredictClassifier())

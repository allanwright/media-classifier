'''Defines a pipeline step which makes a prediction with the classification model.

'''

from src import classifier
from src.step import Step

class PredictClassifier(Step):
    '''Defines a pipeline step which makes a prediction with the classification model.

    '''

    def __init__(self):
        super(PredictClassifier, self).__init__()
        self.input = {
        }
        self.output = {
        }

    def run(self):
        '''Runs the pipeline step.

        '''
        return classifier.predict(self.input['--filename'])

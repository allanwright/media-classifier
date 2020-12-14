'''Defines a pipeline step which makes a prediction with the classification model.

'''

from src import classifier
from src.step import Step

class PredictClassifier(Step):
    '''Defines a pipeline step which makes a prediction with the classification model.

    '''

    def run(self):
        '''Runs the pipeline step.

        '''
        return classifier.predict(self.input['--filename'])

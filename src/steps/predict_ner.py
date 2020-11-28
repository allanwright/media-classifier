'''Defines a pipeline step which makes a prediction with the ner model.

'''

from src import ner
from src.step import Step

class PredictNer(Step):
    '''Defines a pipeline step which makes a prediction with the ner model.

    '''

    def __init__(self):
        super(PredictNer, self).__init__()
        self.input = {
        }
        self.output = {
        }

    def run(self):
        '''Runs the pipeline step.

        '''
        return ner.predict_and_print(self.input['--filename'])

'''Defines a pipeline step which makes a prediction with the ner model.

'''

from src import ner
from src.step import Step

class PredictNer(Step):
    '''Defines a pipeline step which makes a prediction with the ner model.

    '''

    def run(self):
        '''Runs the pipeline step.

        '''
        return ner.predict_and_print(self.input['--filename'])

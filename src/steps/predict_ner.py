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
        filename = self.input['--filename']
        model_id = self.input['--id']
        self.print('Using model \'{model_id}\'', model_id=model_id)
        return ner.predict_and_print(filename, model_id)

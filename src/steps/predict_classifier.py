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
        filename = self.input['--filename']
        model_id = self.input['--id']
        self.print('Using model \'{model_id}\'', model_id=model_id)
        return classifier.predict_and_print(filename, model_id)

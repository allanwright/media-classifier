'''Defines a pipeline step which makes a prediction with the ner model.

'''

from mccore import EntityRecognizer
from mccore import ner
from mccore import persistence

from src.step import Step

class PredictNer(Step):
    '''Defines a pipeline step which makes a prediction with the ner model.

    '''

    def __init__(self):
        super(PredictNer, self).__init__()
        self.input = {
            'model': 'models/ner_mdl.pickle'
        }
        self.output = {
        }

    def run(self):
        '''Runs the pipeline step.

        '''
        nlp, _ = ner.get_model()
        nlp_bytes = persistence.bin_to_obj(self.input['model'])
        nlp.from_bytes(nlp_bytes)
        recognizer = EntityRecognizer(nlp)
        print(recognizer.predict(self.input['--filename']))

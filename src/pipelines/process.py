'''Defines a processing pipeline that prepares data for training.

'''

from src.pipeline import Pipeline
from src.steps.merge import Merge
from src.steps.prepare_classification_data import PrepareClassificationData
from src.steps.prepare_ner_data import PrepareNerData

class Process(Pipeline):
    '''Defines a processing pipeline that prepares data for training.

    '''

    def __init__(self, args):
        '''Initializes a new instance of the Process object.

        Args:
            args (dict): Dictionary of arguments that can be passed as input to every step.
        '''
        super(Process, self).__init__(args)
        if '--model' not in args:
            self.add_step(Merge())
            self.add_step(PrepareClassificationData())
            self.add_step(PrepareNerData())
        elif args['--model'] == 'ner':
            self.add_step(PrepareNerData())
        elif args['--model'] == 'classifier':
            self.add_step(PrepareClassificationData())

'''Defines a processing pipeline that trains a model.

'''

from src.pipeline import Pipeline
from src.steps.train_classifier import TrainClassifier as TrainClassifierStep
from src.steps.train_ner_model import TrainNerModel
from src.steps.validate_classifier import ValidateClassifier
from src.steps.validate_ner import ValidateNer

class Train(Pipeline):
    '''Defines a processing pipeline that trains a model.

    '''

    def __init__(self, args):
        '''Initializes a new instance of the Train object.

        Args:
            args (dict): Dictionary of arguments that can be passed as input to every step.
        '''
        super(Train, self).__init__(args)
        if args['--model'] == 'ner':
            self.add_step(TrainNerModel())
            self.add_step(ValidateNer())
        elif args['--model'] == 'classifier':
            self.add_step(TrainClassifierStep())
            self.add_step(ValidateClassifier())

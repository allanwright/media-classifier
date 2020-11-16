'''Defines a processing pipeline that does all the things.

'''

from src.pipeline import Pipeline
from src.steps.get_test_data import GetTestData
from src.steps.merge import Merge
from src.steps.prepare_classification_data import PrepareClassificationData
from src.steps.prepare_ner_data import PrepareNerData
from src.steps.train_classifier import TrainClassifier
from src.steps.validate_classifier import ValidateClassifier
from src.steps.train_ner_model import TrainNerModel

class AllTheThings(Pipeline):
    '''Defines a processing pipeline that does all the things.

    '''

    def __init__(self, args):
        '''Initializes a new instance of the AllTheThings object.

        Args:
            args (dict): Dictionary of arguments that can be passed as input to every step.
        '''
        super(AllTheThings, self).__init__(args)
        self.add_step(GetTestData())
        self.add_step(Merge())
        self.add_step(PrepareClassificationData())
        self.add_step(PrepareNerData())
        self.add_step(TrainClassifier())
        self.add_step(ValidateClassifier())
        self.add_step(TrainNerModel())

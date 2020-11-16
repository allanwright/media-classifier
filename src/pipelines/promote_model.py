'''Defines a processing pipeline that promotes a model to production.

'''

from src.pipeline import Pipeline
from src.steps.upload_model import UploadModel

class PromoteModel(Pipeline):
    '''Defines a processing pipeline that promotes a model to production.

    '''

    def __init__(self, args):
        '''Initializes a new instance of the PromoteModel object.

        Args:
            args (dict): Dictionary of arguments that can be passed as input to every step.
        '''
        super(PromoteModel, self).__init__(args)
        self.add_step(UploadModel())

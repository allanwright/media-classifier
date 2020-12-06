'''Defines a processing pipeline that promotes a model to production.

'''

from src.pipeline import Pipeline
from src.steps.upload_model import UploadModel

class Promote(Pipeline):
    '''Defines a processing pipeline that promotes a model to production.

    '''

    def __init__(self, args):
        '''Initializes a new instance of the Promote object.

        Args:
            args (dict): Dictionary of arguments that can be passed as input to every step.
        '''
        super(Promote, self).__init__(args)
        self.add_step(UploadModel())

'''Defines a processing pipeline that promotes a model to production.

'''

from src.pipeline import Pipeline
from src.steps.upload_model import UploadModel

class PromoteModel(Pipeline):
    '''Defines a processing pipeline that promotes a model to production.

    '''

    def __init__(self):
        '''Initializes a new instance of the PromoteModel object.

        '''
        super(PromoteModel, self).__init__()
        self.add_step(UploadModel())

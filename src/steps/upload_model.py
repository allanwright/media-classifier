'''Defines a pipeline step which uploads a model to the production environment.

'''

from src.step import Step

class UploadModel(Step):
    '''Defines a pipeline step which uploads a model to the production environment.

    '''

    def __init__(self):
        '''Initializes a new instance of the UploadModel object.

        '''
        super(UploadModel, self).__init__()
        self.input = {
        }
        self.output = {
        }

    def run(self):
        '''Runs the pipeline step.

        '''
        print('Do the work.')

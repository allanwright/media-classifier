'''Defines a pipeline step which promotes a classification model to production.

'''

from src.step import Step

class PromoteClassifier(Step):
    '''Defines a pipeline step which promotes a classification model to production.

    '''

    def __init__(self):
        '''Initializes a new instance of the PromoteClassifier object.

        '''
        super(PromoteClassifier, self).__init__()
        self.input = {
        }
        self.output = {
        }

    def run(self):
        '''Runs the pipeline step.

        '''
        print('Do the work.')

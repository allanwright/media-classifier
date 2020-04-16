'''Defines a processing pipeline that prepares data for training.

'''

from src.pipeline import Pipeline
from src.steps.merge import Merge
from src.steps.prepare import Prepare

class Process(Pipeline):
    '''Defines a processing pipeline that prepares data for training.

    '''

    def __init__(self):
        '''Initializes a new instance of the Process object.

        '''
        super(Process, self).__init__()
        self.add_step(Merge())
        self.add_step(Prepare())

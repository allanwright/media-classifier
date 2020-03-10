from src.Pipeline import Pipeline
from src.steps.Merge import Merge

class Process(Pipeline):
    ''' Defines a processing pipeline that prepares data for training.

    '''

    def __init__(self):
        ''' Initializes a new instance of the Process object.

        '''
        super(Process, self).__init__()
        self.add_step(Merge())
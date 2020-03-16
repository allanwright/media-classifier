''' Defines the interface for a data pipeline step.

'''

from abc import abstractmethod

class Step():
    ''' Defines the interface for a data pipeline step.

    '''

    @abstractmethod
    def run(self):
        ''' Runs the pipeline step.

        '''

from abc import abstractmethod

class IPipelineStep():
    ''' Defines the interface for a data pipeline step.
    '''

    @abstractmethod
    def run(self):
        ''' Runs the pipeline step.
        '''
        pass
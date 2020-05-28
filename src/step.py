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

    def print(self, message: str, **kwargs):
        '''Prints a message about the progress of the pipeline step.

        Args:
            message (string): The message to print.
            kwargs (args): The arguments used to format the message.
        '''
        step_name = self.__class__.__name__
        message = message.format(**kwargs)
        print('Step \'{step_name}\': {message}'.format(
            step_name=step_name,
            message=message))

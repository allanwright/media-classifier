''' Defines a data processing pipeline.

'''

class Pipeline():
    ''' Defines a data processing pipeline.

    '''

    def __init__(self):
        ''' Initializes a new instance of the Pipeline object.

        '''
        self.steps = []

    def add_step(self, step):
        ''' Adds a step to the pipeline.

        '''
        self.steps.append(step)

    def run(self):
        ''' Runs the pipeline.

        '''
        for step in self.steps:
            step.run()

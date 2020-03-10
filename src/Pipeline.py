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
            step_input = step.get_input()
            step_output = step.get_output()
            step.run(step_input, step_output)
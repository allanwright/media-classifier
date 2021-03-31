'''Defines a data processing pipeline.

'''

class Pipeline():
    '''Defines a data processing pipeline.

    '''

    def __init__(self, args):
        '''Initializes a new instance of the Pipeline object.

        Args:
            args (dict): Dictionary of arguments that can be passed as input to every step.
        '''
        self.args = args
        self.steps = []

    def add_step(self, step):
        '''Adds a step to the pipeline.

        '''
        step.input.update(self.args)
        self.steps.append(step)

    def run(self):
        '''Runs the pipeline.

        '''
        output = {}
        for step in self.steps:
            step.input.update(output)
            print(step.input)
            step.run()
            output.update(self.__get_step_output(step))

    def __get_step_output(self, step):
        step_name = step.__class__.__name__
        return {step_name + '_' + k: v for (k, v) in step.output.items()}

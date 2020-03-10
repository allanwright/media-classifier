from abc import abstractmethod

class IPipelineStep():
    ''' Defines the interface for a data pipeline step.

    A data pipeline step represents a unit of work that is performed on a set of
    input data and produces a set of output data. The input and output data is
    typically file based.

    When pipeline steps are added to and run as part of a pipeline, the input
    and output is used to determine if the step needs to run. This is achieved
    by hashing the input and output and recording the hashes with each run, in
    addition to hashing the python file containing the code that runs the
    pipeline step, in order to determine when a step needs to run as part of a
    pipeline. When code changes in a python file other than the pipeline step
    file but is called from it, a way to tell the pipeline to force a step to
    run will be provided.

    '''

    @abstractmethod
    def get_input(self):
        ''' Gets the input for the pipeline step.

        The input of a pipeline step is expected to either be completely file
        based or non file based, such as downloading html documents from a
        remote server or processing files on local storage. It is expected that
        a pipeline step will not contain inputs of different types. If it does,
        you are doing this wrong.

        Returns:
            list: The list of inputs.
        '''
        pass

    @abstractmethod
    def get_output(self):
        ''' Gets the output for the pipeline step.

        The output of a pipeline step is expected to either be completely file
        based or non file based, such as uploading documents to a storage queue
        or writing files to local storage. It is expected that a pipeline step
        will not contain inputs of different types. If it does, you are doing
        this wrong.

        Returns:
            list: The list of outputs.
        '''
        pass

    @abstractmethod
    def run(self):
        ''' Runs the pipeline step.

        '''
        pass
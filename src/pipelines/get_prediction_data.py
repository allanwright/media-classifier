'''Defines a processing pipeline that aquires prediction data.

'''

from src.pipeline import Pipeline
from src.steps.get_prediction_data import GetPredictionData as GetPredictionDataStep

class GetPredictionData(Pipeline):
    '''Defines a processing pipeline that aquires prediction data.

    '''

    def __init__(self):
        '''Initializes a new instance of the GetPredictionData object.

        '''
        super(GetPredictionData, self).__init__()
        self.add_step(GetPredictionDataStep())

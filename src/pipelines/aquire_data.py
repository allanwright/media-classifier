'''Defines a processing pipeline that aquires training and test data.

'''

from src.pipeline import Pipeline
from src.steps.get_prediction_data import GetPredictionData
#from src.steps.get_kraken_data import GetKrakenData

class AquireData(Pipeline):
    '''Defines a processing pipeline that aquires training and test data.

    '''

    def __init__(self):
        '''Initializes a new instance of the AquireData object.

        '''
        super(AquireData, self).__init__()
        #self.add_step(GetKrakenData())
        self.add_step(GetPredictionData())

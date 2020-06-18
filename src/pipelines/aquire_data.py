'''Defines a processing pipeline that aquires training and test data.

'''

from src.pipeline import Pipeline
from src.steps.get_test_data import GetTestData
#from src.steps.get_kraken_data import GetKrakenData
#from src.steps.get_pig_data import GetPigData
#from src.steps.get_xerus_data import GetXerusData
#from src.steps.get_yak_data import GetYakData

class AquireData(Pipeline):
    '''Defines a processing pipeline that aquires training and test data.

    '''

    def __init__(self):
        '''Initializes a new instance of the AquireData object.

        '''
        super(AquireData, self).__init__()
        #self.add_step(GetKrakenData())
        #self.add_step(GetPigData())
        #self.add_step(GetXerusData())
        #self.add_step(GetYakData())
        self.add_step(GetTestData())

'''Defines a processing pipeline that aquires training data.

'''

from src.pipeline import Pipeline
#from src.steps.get_kraken_data import GetKrakenData
#from src.steps.get_pig_data import GetPigData
from src.steps.get_xerus_data import GetXerusData
from src.steps.get_yak_data import GetYakData

class AquireTrainData(Pipeline):
    '''Defines a processing pipeline that aquires training data.

    '''

    def __init__(self, args):
        '''Initializes a new instance of the AquireTrainData object.

        Args:
            args (dict): Dictionary of arguments that can be passed as input to every step.
        '''
        super(AquireTrainData, self).__init__(args)
        #self.add_step(GetKrakenData())
        #self.add_step(GetPigData())
        self.add_step(GetXerusData())
        self.add_step(GetYakData())
'''Defines a processing pipeline that aquires training data.

'''

from src.pipeline import Pipeline
from src.steps.get_kraken_data import GetKrakenData
from src.steps.get_pig_data import GetPigData
from src.steps.get_test_data import GetTestData
from src.steps.get_xerus_data import GetXerusData
from src.steps.get_yak_data import GetYakData

class Aquire(Pipeline):
    '''Defines a processing pipeline that aquires training or test data.

    '''

    def __init__(self, args):
        '''Initializes a new instance of the Aquire object.

        Args:
            args (dict): Dictionary of arguments that can be passed as input to every step.
        '''
        super(Aquire, self).__init__(args)
        if args['--source'] == 'kraken':
            self.add_step(GetKrakenData())
        elif args['--source'] == 'pig':
            self.add_step(GetPigData())
        elif args['--source'] == 'test':
            self.add_step(GetTestData())
        elif args['--source'] == 'yak':
            self.add_step(GetYakData())
        elif args['--source'] == 'xerus':
            self.add_step(GetXerusData())

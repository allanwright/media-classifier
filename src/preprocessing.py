'''Aquisition of data for training.

'''

from src.pipelines.process import Process
from src.steps.merge import Merge
from src.steps.prepare_classification_data import PrepareClassificationData

def process_all():
    '''Performs all data processing steps.

    '''
    pipeline = Process()
    pipeline.run()

def process_merge():
    '''Performs merging of data.

    '''
    merge = Merge()
    merge.run()

def process_feature():
    '''Performs feature generation.

    '''
    prepare = PrepareClassificationData()
    prepare.run()

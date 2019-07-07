'''Media-Classifier CLI

Usage:
    mc aquire <source>
    mc process
    mc train <model>
    mc eval

Arguments:
    <source>    Source to aquire data from (pig, kraken, xerus, yak)
    <model>     Model to train (baseline)

'''
import os
from docopt import docopt
from dotenv import load_dotenv
from src import dataset, train

def main():
    BASE_DIR = os.path.abspath(os.path.dirname(__file__))
    load_dotenv(dotenv_path=os.path.join(BASE_DIR, ".env"))
    arguments = docopt(__doc__)

    if arguments['aquire']:
        source = arguments['<source>']
        if source == 'kraken':
            dataset.get_kraken_data(os.getenv('KRAKEN_PATH'))
        elif source == 'pig':
            dataset.get_pig_data(os.getenv('PIG_PATH'))
        elif source == 'xerus':
            dataset.get_xerus_data(os.getenv('XERUS_URL'))
        elif source == 'yak':
            dataset.get_yak_data(os.getenv('YAK_URL'))
        else:
            print('Invalid source')
    
    if arguments['process']:
        dataset.process_data()
    
    if arguments['train']:
        model = arguments['<model>']
        if model == 'baseline':
            train.train_baseline()

if __name__ == '__main__':
    main()
'''Media-Classifier CLI

Usage:
    mc aquire <source>
    mc process
    mc train <model>
    mc eval <model> <filename>

Arguments:
    <source>    Source to aquire data from (pig, kraken, xerus, yak)
    <model>     Model to train (baseline)
    <filename>  The filename to evaluate
'''
import os
from docopt import docopt
from dotenv import load_dotenv
from src import dataset, train, inference

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
    elif arguments['process']:
        dataset.process_data()    
    elif arguments['train']:
        model = arguments['<model>']
        if model == 'baseline':
            train.train_baseline()
    elif arguments['eval']:
        model = arguments['<model>']
        filename = arguments['<filename>']
        if model == 'baseline':
            inference.eval_baseline_model(filename)

if __name__ == '__main__':
    main()
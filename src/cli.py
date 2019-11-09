'''Media-Classifier CLI

Usage:
    mc aquire <source>
    mc clean
    mc process
    mc train <model>
    mc predict <model> <filename>

Arguments:
    <source>    Source to aquire data from (pig, kraken, xerus, yak)
    <model>     Model to train/predict (baseline, cnn, ner)
    <filename>  The filename to evaluate
'''
import os
from docopt import docopt
from dotenv import load_dotenv
from src import datasets, preprocessing
from src.models import baseline, cnn, ner

def main():
    BASE_DIR = os.path.abspath(os.path.dirname(__file__))
    load_dotenv(dotenv_path=os.path.join(BASE_DIR, ".env"))
    arguments = docopt(__doc__)

    if arguments['aquire']:
        source = arguments['<source>']
        if source == 'kraken':
            datasets.get_kraken_data(os.getenv('KRAKEN_PATH'))
        elif source == 'pig':
            datasets.get_pig_data(os.getenv('PIG_PATH'))
        elif source == 'xerus':
            datasets.get_xerus_data(os.getenv('XERUS_URL'))
        elif source == 'yak':
            datasets.get_yak_data(os.getenv('YAK_URL'))
        else:
            print('Invalid source')
    elif arguments['process']:
        preprocessing.process_data()
    elif arguments['train']:
        model = arguments['<model>']
        if model == 'baseline':
            baseline.train()
        elif model == 'cnn':
            cnn.train()
        elif model == 'ner':
            ner.train()
    elif arguments['predict']:
        model = arguments['<model>']
        filename = arguments['<filename>']
        if model == 'baseline':
            baseline.predict(filename)
        elif model == 'cnn':
            cnn.predict(filename)
        elif model == 'ner':
            ner.predict(filename)

if __name__ == '__main__':
    main()
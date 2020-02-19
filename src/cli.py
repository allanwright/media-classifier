'''Media-Classifier CLI

Usage:
    mc aquire <source>
    mc clean
    mc process
    mc train <model>
    mc predict <model> <filename>

Arguments:
    <source>    Source to aquire data from (pig, kraken, xerus, yak, prediction)
    <model>     Model to train/predict (baseline, cnn, ner)
    <filename>  The filename to evaluate
'''
import os
from docopt import docopt
from dotenv import load_dotenv
from src import datasets, preprocessing
from src.models import baseline, cnn, ner

def main():
    load_dotenv()
    args = docopt(__doc__)

    if args['aquire']:
        source = args['<source>']
        if source == 'kraken':
            datasets.get_kraken_data(os.getenv('KRAKEN_PATH'))
        elif source == 'pig':
            datasets.get_pig_data(os.getenv('PIG_PATH'))
        elif source == 'xerus':
            datasets.get_xerus_data(os.getenv('XERUS_URL'))
        elif source == 'yak':
            datasets.get_yak_data(os.getenv('YAK_URL'))
        elif source == 'predictions':
            datasets.get_prediction_data()
        else:
            print('Invalid source')
    elif args['process']:
        preprocessing.process_data()
    elif args['train']:
        resolve_method(args['<model>'], 'train')()
    elif args['predict']:
        model = args['<model>']
        filename = args['<filename>']
        if model == 'baseline':
            baseline.predict(filename)
        elif model == 'cnn':
            cnn.predict(filename)
        elif model == 'ner':
            ner.predict(filename)

def resolve_method(module, method):
    return getattr(globals()[module], method)

if __name__ == '__main__':
    main()
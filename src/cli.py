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
        if source == 'prediction':
            resolve_method(datasets, f'get_{source}_data')()
        else:
            path = os.getenv(f'{source}_PATH')
            resolve_method(datasets, f'get_{source}_data')(path)
    elif args['process']:
        preprocessing.process_data()
    elif args['train']:
        resolve_method(args['<model>'], 'train')()
    elif args['predict']:
        resolve_method(args['<model>'], 'predict')(args['<filename>'])

def resolve_method(module, method):
    if isinstance(module, str):
        return getattr(globals()[module], method)
    else:
        return getattr(module, method)

if __name__ == '__main__':
    main()
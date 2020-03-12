'''Media-Classifier CLI

Usage:
    mc aquire <source>
    mc process <step>
    mc train <model>
    mc predict <model> <filename>

Arguments:
    <source>    Source to aquire data from (pig, kraken, xerus, yak, prediction)
    <step>      Processing step to run (all, merge, feature)
    <model>     Model to train/predict (baseline, cnn, ner)
    <filename>  The filename to evaluate
'''

import os

from docopt import docopt
from dotenv import load_dotenv

from src import datasets, preprocessing

def main():
    ''' The entry point of the package.

    '''
    load_dotenv()
    args = docopt(__doc__)

    if args['aquire']:
        source = args['<source>']
        if source == 'prediction':
            resolve_method(datasets, f'get_{source}_data')()
        else:
            path = os.getenv(f'{source.upper()}_URL')
            resolve_method(datasets, f'get_{source}_data')(path)
    elif args['process']:
        step = args['<step>']
        resolve_method(preprocessing, f'process_{step}')()
    elif args['train']:
        resolve_method(args['<model>'], 'train')()
    elif args['predict']:
        resolve_method(args['<model>'], 'predict')(args['<filename>'])

def resolve_method(module, method):
    ''' Resolves a method from the specified module and method name.

    Args:
        module (module or string): The module or the name of the module to resolve the method for.
        method (string): The name of the method to resolve.

    Returns:
        method: The method.
    '''
    if isinstance(module, str):
        return getattr(globals()[module], method)
    else:
        return getattr(module, method)

if __name__ == '__main__':
    main()

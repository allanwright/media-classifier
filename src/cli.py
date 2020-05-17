'''Media-Classifier CLI

Usage:
    mc run <pipeline>
    mc train <model>
    mc predict <model> <filename>

Arguments:
    <pipeline>      The name of the pipeline to run
    <model>         Model to train/predict (baseline, cnn, ner)
    <filename>      The filename to evaluate

Pipelines:
    <aquire-data>   Aquires training and test data
    <process-data>  Processes training and test data
'''

from docopt import docopt
from dotenv import load_dotenv

# pylint: disable=unused-import
from src.models import baseline, cnn, ner
from src.pipelines.aquire_data import AquireData as aquire_data
from src.pipelines.process_data import ProcessData as process_data

def main():
    '''The entry point of the package.

    '''
    load_dotenv()
    args = docopt(__doc__)

    if args['run']:
        pipeline = __resolve_class(args['<pipeline>'])()
        pipeline.run()
    elif args['train']:
        __resolve_method(args['<model>'], 'train')()
    elif args['predict']:
        __resolve_method(args['<model>'], 'predict')(args['<filename>'])

def __resolve_class(name: str):
    return globals()[name.replace('-', '_')]

def __resolve_method(module, method):
    if isinstance(module, str):
        return getattr(globals()[module], method)

    return getattr(module, method)

if __name__ == '__main__':
    main()

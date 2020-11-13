'''Media-Classifier CLI

Usage:
    mc run <pipeline>
    mc predict <model> <filename>

Arguments:
    <pipeline>              The name of the pipeline to run
    <model>                 Model to use for predictions (classifier, ner)
    <filename>              The filename to evaluate

Pipelines:
    aquire-train-data       Aquires training data
    aquire-test-data        Aquires test data
    process-data            Merges and processes training and test data for all models
    process-classifier      Process training and test data used by the classification model
    process-ner             Process training and test data used by the ner model
    train-classifier        Trains the classification model
    train-ner               Trains the ner model
    all-the-things          Does all the things
    promote-classifier      Promotes a classification model to production
'''

from docopt import docopt
from dotenv import load_dotenv

# pylint: disable=unused-import
from src import classifier, ner
from src.pipelines.aquire_train_data import AquireTrainData as aquire_train_data
from src.pipelines.aquire_test_data import AquireTestData as aquire_test_data
from src.pipelines.process_data import ProcessData as process_data
from src.pipelines.process_classifier import ProcessClassifier as process_classifier
from src.pipelines.process_ner import ProcessNer as process_ner
from src.pipelines.train_classifier import TrainClassifier as train_classifier
from src.pipelines.train_ner import TrainNer as train_ner
from src.pipelines.all_the_things import AllTheThings as all_the_things
from src.pipelines.promote_classifier import PromoteClassifier as promote_classifier

def main():
    '''The entry point of the package.

    '''
    load_dotenv()
    args = docopt(__doc__)

    if args['run']:
        pipeline = __resolve_class(args['<pipeline>'])()
        pipeline.run()
    elif args['predict']:
        __resolve_method(args['<model>'], 'predict_and_print')(args['<filename>'])

def __resolve_class(name: str):
    return globals()[name.replace('-', '_')]

def __resolve_method(module, method):
    if isinstance(module, str):
        return getattr(globals()[module], method)

    return getattr(module, method)

if __name__ == '__main__':
    main()

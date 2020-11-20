'''Media-Classifier CLI

Usage:
    mc aquire-train-data
    mc aquire-test-data
    mc process-data
    mc process-classifier
    mc process-ner
    mc train-classifier
    mc train-ner
    mc promote-model --model <model>
    mc predict <model> <filename>

Arguments:
    -m <model>, --model <model>     Type of model (classifier, ner)
    <filename>                      The filename to evaluate

Pipelines:
    aquire-train-data       Aquires training data
    aquire-test-data        Aquires test data
    process-data            Merges and processes training and test data for all models
    process-classifier      Process training and test data used by the classification model
    process-ner             Process training and test data used by the ner model
    train-classifier        Trains the classification model
    train-ner               Trains the ner model
    promote-model           Promotes a model to production
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
from src.pipelines.promote_model import PromoteModel as promote_model

def main():
    '''The entry point of the package.

    '''
    load_dotenv()
    args = docopt(__doc__)

    if args['predict']:
        __resolve_method(args['<model>'], 'predict_and_print')(args['<filename>'])
    else:
        __run_pipeline(args)

def __get_first_true_command(args):
    return [k for k, v in args.items() if v][0]

def __resolve_pipeline(name):
    return globals()[name.replace('-', '_')]

def __run_pipeline(args):
    pipeline_name = __get_first_true_command(args)
    pipeline = __resolve_pipeline(pipeline_name)()
    pipeline.run()

def __resolve_method(module, method):
    if isinstance(module, str):
        return getattr(globals()[module], method)

    return getattr(module, method)

if __name__ == '__main__':
    main()

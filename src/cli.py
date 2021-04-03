'''Media-Classifier CLI

Usage:
    mc aquire-train-data
    mc aquire-test-data
    mc process [--model <model>]
    mc train --model <model>
    mc promote --model <model> --id <model>
    mc predict --model <model> --filename <filename> [--id <id>]

Arguments:
    -m <model>, --model <model>             Type of model (classifier, ner)
    -i <id>, --id <id>                      The model id [default: latest]
    -f <filename>, --filename <filename>    The filename to evaluate

Pipelines:
    aquire-train-data       Aquires training data
    aquire-test-data        Aquires test data
    process                 Processes training and test data
    train                   Trains a model
    train-ner               Trains the ner model
    promote                 Promotes a model to production
    predict                 Uses a model to make a prediction
'''

from docopt import docopt
from dotenv import load_dotenv

# pylint: disable=unused-import
from src.pipelines.aquire_train_data import AquireTrainData as aquire_train_data
from src.pipelines.aquire_test_data import AquireTestData as aquire_test_data
from src.pipelines.process import Process as process
from src.pipelines.train import Train as train
from src.pipelines.promote import Promote as promote
from src.pipelines.predict import Predict as predict

def main():
    '''The entry point of the package.

    '''
    load_dotenv()
    args = docopt(__doc__)
    pipeline_name = [k for k, v in args.items() if v is True][0]
    pipeline_args = {k:v for k, v in args.items() if k.startswith('--') and v is not None}
    pipeline = globals()[pipeline_name.replace('-', '_')](pipeline_args)
    pipeline.run()

if __name__ == '__main__':
    main()

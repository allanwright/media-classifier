'''Media-Classifier CLI

Usage:
    mc aquire --source <source>
    mc predict --model <model> --filename <filename> [--id <id>]
    mc process [--model <model>]
    mc promote --model <model> --id <model>
    mc train --model <model>

Arguments:
    -s <source>, --source <source>          Data source (kraken, pig, test, xerus, yak)
    -m <model>, --model <model>             Type of model (classifier, ner)
    -i <id>, --id <id>                      The model id [default: latest]
    -f <filename>, --filename <filename>    The filename to evaluate

Pipelines:
    aquire                  Aquires training or test data
    predict                 Uses a model to make a prediction
    process                 Processes training and test data
    promote                 Promotes a model to production
    train                   Trains a model
'''

from docopt import docopt
from dotenv import load_dotenv

# pylint: disable=unused-import
from src.pipelines.aquire import Aquire as aquire
from src.pipelines.predict import Predict as predict
from src.pipelines.process import Process as process
from src.pipelines.promote import Promote as promote
from src.pipelines.train import Train as train

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

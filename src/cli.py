'''Media-Classifier CLI

Usage:
    mc-cli aquire <source>
    mc-cli process
    mc-cli train
    mc-cli eval

Arguments:
    <source>    Source to aquire data from (pig, kraken, xerus, yak)

'''
import os
from docopt import docopt
from dotenv import load_dotenv
from src import dataset

def main():
    load_dotenv()
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

    """ if arguments['train']:
        train_model(arguments['<dataset-dir>'],
                    arguments['<model-file>'],
                    int(arguments['--vocab-size'])
        )
    elif arguments['ask']:
        ask_model(arguments['<model-file>'],
                  arguments['<question>']) """

if __name__ == '__main__':
    main()
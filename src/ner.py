'''Module that wraps the Ner class for the purpose of using the model
to make predictions.

'''

import os

from mccore import EntityRecognizer
from mccore import ner
from mccore import persistence

def predict(filename, model_id='latest'):
    ''' Makes a prediction using the named entity recognition model.

    Args:
        input (filename): The filename to evaluate.
        model_id (string): the id of the model to use.
    '''
    model_path = __get_model_path(model_id)
    nlp, _ = ner.get_model()
    nlp_bytes = persistence.bin_to_obj(model_path + 'ner_mdl.pickle')
    nlp.from_bytes(nlp_bytes)
    recognizer = EntityRecognizer(nlp)
    return recognizer.predict(filename)

def predict_and_print(filename, model_id='latest'):
    ''' Makes a prediction using the named entity recognition model and prints the result.

    Args:
        input (filename): The filename to evaluate.
        model_id (string): the id of the model to use.
    '''
    result = predict(filename, model_id)
    print(result)

def __get_model_path(model_id: str):
    base_path = 'models/ner'
    if model_id.lower() == 'latest':
        model_id = sorted(os.listdir(base_path), reverse=True)[0]

    return '{base_path}/{model_id}/'.format(
            base_path=base_path,
            model_id=model_id)

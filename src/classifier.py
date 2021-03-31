'''Module that wraps the Classifier class for the purpose of using the model
to make predictions.

'''

import os

from mccore import persistence
from mccore import Classifier

def predict(filename, model_id='latest'):
    ''' Makes a prediction using the classification model.

    Args:
        filenanme (string): The filename to evaluate.
        model_id (string): the id of the model to use.
    '''
    model_path = __get_model_path(model_id)
    classifier = Classifier(
        persistence.bin_to_obj(model_path + 'classifier_vec.pickle'),
        persistence.bin_to_obj(model_path + 'classifier_mdl.pickle'),
        persistence.json_to_obj('data/processed/label_dictionary.json')
    )
    classification = classifier.predict(filename)

    return (classification['label'], classification['probability'])

def predict_and_print(filename, model_id='latest'):
    ''' Makes a prediction using the classification model and prints the result.

    Args:
        filenanme (string): The filename to evaluate.
        model_id (string): The id of the model to use.
    '''
    label, confidence = predict(filename, model_id)
    print('Predicted class \'{label}\' with {confidence:.2f}% confidence.'
          .format(label=label, confidence=confidence*100))

def __get_model_path(model_id: str):
    base_path = 'models/classifier'
    if model_id.lower() == 'latest':
        model_id = sorted(os.listdir(base_path), reverse=True)[0]

    return '{base_path}/{model_id}/'.format(
            base_path=base_path,
            model_id=model_id)

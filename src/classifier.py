'''Module that wraps the Classifier class for the purpose of using the model
to make predictions.

'''

from mccore import persistence
from mccore import Classifier

def predict(filename):
    ''' Makes a prediction using the classification model.

    Args:
        filenanme (string): The filename to evaluate.
    '''
    classifier = Classifier(
        persistence.bin_to_obj('models/classifier_vec.pickle'),
        persistence.bin_to_obj('models/classifier_mdl.pickle'),
        persistence.json_to_obj('data/processed/label_dictionary.json')
    )
    label, confidence = classifier.predict(filename)

    return (label, confidence)

def predict_and_print(filename):
    ''' Makes a prediction using the classification model and prints the result.

    Args:
        filenanme (string): The filename to evaluate.
    '''
    label, confidence = predict(filename)
    print('Predicted class \'{label}\' with {confidence:.2f}% confidence.'
          .format(label=label, confidence=confidence*100))

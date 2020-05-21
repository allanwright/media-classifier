'''Module that wraps the Ner class for the purpose of using the model
to make predictions.

'''

from mccore import EntityRecognizer
from mccore import ner
from mccore import persistence

def predict(filename):
    ''' Makes a prediction using the named entity recognition model.

    Args:
        input (filename): The filename to evaluate.
    '''
    nlp, _ = ner.get_model()
    nlp_bytes = persistence.bin_to_obj('models/ner_mdl.pickle')
    nlp.from_bytes(nlp_bytes)
    recognizer = EntityRecognizer(nlp)
    print(recognizer.predict(filename))

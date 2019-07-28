import json
import numpy as np

def get_label(proba):
    '''Gets the label from the specified class probability estimates.

    Args:
        proba (array like): The estimated class probability estimates.
    
    Returns:
        label (string): The label associated with the class having the highest probability estimate.
        esimate (float): The probability estimate.
    '''
    label_json = ''
    with open('data/processed/label_encoding.json', 'r') as json_file:
        label_json = json_file.read()    
    labels = json.loads(label_json)
    return (labels[str(np.argmax(proba))], np.max(proba))
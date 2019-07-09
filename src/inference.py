import numpy as np
from joblib import load

def eval_baseline_model(filename):
    ''' Evaluates the baseline model.

    Args:
        input (filename): The filename to evaluate.
    '''
    vectorizer = load('models/baseline_vectorizer.joblib')
    x = process_input(filename)
    x = vectorizer.transform(np.array([x]))
    classifier = load('models/baseline_model.joblib')
    y = classifier.predict_proba(x)
    np.set_printoptions(suppress=True)
    print(y)

    y = np.argmax(y)

    if y == 0:
        print('app')
    elif y == 1:
        print('movie')
    elif y == 2:
        print('music')
    elif y == 3:
        print('tv')

def process_input(filename):
    # Remove commas
    filename = filename.replace(',', '')

    # Remove file sizes
    filename = filename.replace(r'\s{1}\(.+\)$', '')
    filename = filename.replace(r' - \S+\s{1}\S+$', '')

    # Remove file extension
    filename = filename.replace(r'\.(\w{3})$', '')
    
    # Remove paths
    filename = filename.split('/')[-1]

    # Normalize word separators
    filename = filename.replace('.', ' ')
    filename = filename.replace('_', ' ')
    filename = filename.replace('-', ' ')
    filename = filename.replace('[', ' ')
    filename = filename.replace(']', ' ')
    filename = filename.replace('+', ' ')
    filename = ' '.join(filename.split())

    # Remove rubbish characters
    filename = filename.strip('`~!@#$%^&*()-_+=[]|;:<>,./?')

    return filename
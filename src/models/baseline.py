import numpy as np
import pandas as pd
from joblib import dump, load
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from src import datasets

def train():
    '''Trains the baseline model.

    '''
    x_train, y_train, x_test, y_test = datasets.get_train_test_data()

    labelEncoder = LabelEncoder()
    labelEncoder.fit(y_train)
    y_train = labelEncoder.transform(y_train)
    y_test = labelEncoder.transform(y_test)

    vectorizer = CountVectorizer()
    vectorizer.fit(x_train)
    x_train = vectorizer.transform(x_train)
    x_test = vectorizer.transform(x_test)

    max_iterations = 200

    classifier = LogisticRegression(
        solver='lbfgs', multi_class='multinomial', max_iter=max_iterations)
    
    print('Training baseline model on {sample_count} samples.'.format(
        sample_count=x_train.shape[0]))
    
    classifier.fit(x_train, y_train)

    print('Testing baseline model on {sample_count} samples.'.format(
        sample_count=x_test.shape[0]))

    score = classifier.score(x_test, y_test)

    print('Baseline model accuracy: {accuracy}'.format(accuracy=score))

    dump(vectorizer, 'models/baseline/vectorizer.joblib')
    dump(classifier, 'models/baseline/model.joblib')

    print('Saved model to models/baseline/model.joblib')

def eval(filename):
    ''' Evaluates the baseline model.

    Args:
        input (filename): The filename to evaluate.
    '''
    vectorizer = load('models/baseline/vectorizer.joblib')
    x = process_input(filename)
    x = vectorizer.transform(np.array([x]))
    classifier = load('models/baseline/model.joblib')
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
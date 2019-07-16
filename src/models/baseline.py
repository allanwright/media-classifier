import numpy as np
import pandas as pd
from joblib import dump, load
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression

def train():
    '''Trains the baseline model.

    '''
    x_train = read_x_data('x_train.csv')
    y_train = read_y_data('y_train_ordinal_encoded.csv')
    x_test = read_x_data('x_test.csv')
    y_test = read_y_data('y_test_ordinal_encoded.csv')

    y_train = np.ravel(y_train)
    y_test = np.ravel(y_test)

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

    dump(vectorizer, 'models/baseline_vectorizer.joblib')
    dump(classifier, 'models/baseline_model.joblib')

    print('Saved model to models/baseline.joblib')

def read_x_data(name):
    df = read_data(name)
    df = df['name'].map(str) + ' ' + df['ext']
    return df.to_numpy()

def read_y_data(name):
    return read_data(name).to_numpy()

def read_data(name):
    return pd.read_csv('data/processed/' + name)

def eval(filename):
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
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression

def train_baseline():
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

    classifier = LogisticRegression(solver='lbfgs', multi_class='multinomial', max_iter=200)
    classifier.fit(x_train, y_train)
    score = classifier.score(x_test, y_test)

    print('Accuracy: {accuracy}'.format(accuracy=score))

def read_x_data(name):
    df = read_data(name)
    df = df['name']
    return df.to_numpy()

def read_y_data(name):
    return read_data(name).to_numpy()

def read_data(name):
    return pd.read_csv('data/processed/' + name)
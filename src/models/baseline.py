import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from src import datasets
from src import prediction
from mccore import persistence
from mccore import preprocessing

def train():
    '''Trains the baseline model.

    '''
    x_train, y_train, x_test, y_test = datasets.get_train_test_data()
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

    persistence.obj_to_bin(vectorizer, 'models/cls_base_vec.pickle')
    persistence.obj_to_bin(classifier, 'models/cls_base_mdl.pickle')

def predict(filename):
    ''' Makes a prediction using the baseline model.

    Args:
        filenanme (string): The filename to evaluate.
    '''
    vectorizer = persistence.bin_to_obj('models/cls_base_vec.pickle')
    x = preprocessing.prepare_input(filename)
    x = vectorizer.transform(np.array([x]))
    classifier = persistence.bin_to_obj('models/cls_base_mdl.pickle')
    y = classifier.predict_proba(x)
    np.set_printoptions(suppress=True)
    label, confidence = prediction.get_label(y)
    print('Predicted class \'{label}\' with {confidence:.2f}% confidence.'
        .format(label=label, confidence=confidence*100))
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from src import datasets
from mccore import Classifier
from mccore import persistence
from mccore import prediction
from mccore import preprocessing

def train():
    '''Trains the baseline model.

    '''
    x_train, y_train, x_eval, y_eval, x_test, y_test = datasets.get_train_test_data()
    vectorizer = CountVectorizer()
    vectorizer.fit(x_train)
    x_train = vectorizer.transform(x_train)
    x_eval = vectorizer.transform(x_eval)
    x_test = vectorizer.transform(x_test)

    max_iterations = 200
    classifier = LogisticRegression(
        solver='lbfgs', multi_class='multinomial', max_iter=max_iterations)
    
    print(f'Training baseline model on {x_train.shape[0]} samples.')
    
    classifier.fit(x_train, y_train)
    
    score_model(classifier, x_eval, y_eval)
    score_model(classifier, x_test, y_test)

    persistence.obj_to_bin(vectorizer, 'models/cls_base_vec.pickle')
    persistence.obj_to_bin(classifier, 'models/cls_base_mdl.pickle')

def score_model(classifier, features, labels):
    ''' Scores the accuracy of the baseline model.

    Args:
        classifier (object): The classifier.
        features (array like): The features.
        labels (array like): The labels.
    '''
    score = classifier.score(features, labels)
    print(f'Baseline eval accuracy: {score*100:.2f}% on {features.shape[0]} samples')

def predict(filename):
    ''' Makes a prediction using the baseline model.

    Args:
        filenanme (string): The filename to evaluate.
    '''
    classifier = Classifier(
        persistence.bin_to_obj('models/cls_base_vec.pickle'),
        persistence.bin_to_obj('models/cls_base_mdl.pickle'),
        persistence.json_to_obj('data/processed/label_dictionary.json')
    )
    label, confidence = classifier.predict(filename)

    print('Predicted class \'{label}\' with {confidence:.2f}% confidence.'
        .format(label=label, confidence=confidence*100))
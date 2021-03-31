'''Defines a pipeline step which trains the classification model.

'''

import datetime
import os
import shutil

from mccore import persistence
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import plot_confusion_matrix
import matplotlib.pyplot as plt

from src import datasets
from src.step import Step

class TrainClassifier(Step):
    '''Defines a pipeline step which trains the classification model.

    '''

    def __init__(self):
        super(TrainClassifier, self).__init__()
        self.input = {
        }
        self.output = {
            'results_dir': 'models/classifier/%s/',
            'vectorizer': 'classifier_vec.pickle',
            'model': 'classifier_mdl.pickle',
            'label_dict': 'data/processed/label_dictionary.json',
            'results': 'models/classifier/%s/%s.png',
        }

    def run(self):
        '''Runs the pipeline step.

        '''
        x_train, y_train, x_eval, y_eval, x_test, y_test = datasets.get_train_test_data()
        vectorizer = CountVectorizer()
        vectorizer.fit(x_train)
        x_train = vectorizer.transform(x_train)
        x_eval = vectorizer.transform(x_eval)
        x_test = vectorizer.transform(x_test)
        timestamp = self.__get_timestamp()

        max_iterations = 200
        classifier = LogisticRegression(
            solver='lbfgs', multi_class='multinomial', max_iter=max_iterations)

        self.print('Training classifier with {samples} samples.', samples=x_train.shape[0])

        classifier.fit(x_train, y_train)

        self.output['results_dir'] = self.output['results_dir'] % timestamp
        output_dir = self.output['results_dir']
        if not os.path.exists(output_dir):
            os.mkdir(output_dir)

        self.__score_model(classifier, x_eval, y_eval, 'eval', timestamp)
        self.__score_model(classifier, x_test, y_test, 'test', timestamp)

        persistence.obj_to_bin(vectorizer, output_dir + self.output['vectorizer'])
        persistence.obj_to_bin(classifier, output_dir + self.output['model'])
        shutil.copy(self.output['label_dict'], output_dir)

        print(f'Training complete, check {output_dir} for results.')

    def __score_model(self, classifier, features, labels, title, timestamp):
        classes = persistence.json_to_obj(self.output['label_dict'])
        plot_confusion_matrix(
            estimator=classifier,
            X=features,
            y_true=labels,
            display_labels=classes.values(),
            normalize='true')
        plt.title(title)
        plt.savefig(self.output['results'] % (timestamp, title))

    def __get_timestamp(self):
        return datetime.datetime.today().strftime('%Y%m%dT%H%M%S')

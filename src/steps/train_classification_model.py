'''Defines a pipeline step which trains the classification model.

'''

import datetime

from mccore import persistence
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import plot_confusion_matrix
import matplotlib.pyplot as plt

from src import datasets
from src.step import Step

class TrainClassificationModel(Step):
    '''Defines a pipeline step which trains the classification model.

    '''

    def __init__(self):
        super(TrainClassificationModel, self).__init__()
        self.input = {
        }
        self.output = {
            'vectorizer': 'models/cls_base_vec.pickle',
            'model': 'models/cls_base_mdl.pickle',
            'label_dict': 'data/processed/label_dictionary.json',
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

        max_iterations = 200
        classifier = LogisticRegression(
            solver='lbfgs', multi_class='multinomial', max_iter=max_iterations)

        self.print('Training classifier with {samples} samples.', samples=x_train.shape[0])

        classifier.fit(x_train, y_train)

        self.__score_model(classifier, x_eval, y_eval, 'eval')
        self.__score_model(classifier, x_test, y_test, 'test')

        persistence.obj_to_bin(vectorizer, self.output['vectorizer'])
        persistence.obj_to_bin(classifier, self.output['model'])

        print('Training complete, check /results for model accuracy.')

    def __score_model(self, classifier, features, labels, title):
        classes = persistence.json_to_obj(self.output['label_dict'])
        plot_confusion_matrix(
            estimator=classifier,
            X=features,
            y_true=labels,
            display_labels=classes.values(),
            normalize='true')
        now = datetime.datetime.today().strftime('%Y%m%dT%H%M%S')
        name = f'{now}_{title}'
        plt.title(name)
        plt.savefig(f'results/{name}.png')

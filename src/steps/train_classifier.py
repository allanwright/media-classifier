'''Defines a pipeline step which trains the classification model.

'''

import datetime
import os

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
            'label_dict': 'data/processed/label_dictionary.json',
        }
        self.output = {
            'output_dir': 'models/classifier/{timestamp}',
            'vectorizer': '{output_dir}/classifier_vec.pickle',
            'model': '{output_dir}/classifier_mdl.pickle',
            'eval_results': '{output_dir}/eval.png',
            'test_results': '{output_dir}/test.png'
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

        output_dir = self.output['output_dir'].format(timestamp=timestamp)
        self.output['output_dir'] = output_dir
        if not os.path.exists(output_dir):
            os.mkdir(output_dir)

        self.output['vectorizer'] = self.output['vectorizer'].format(output_dir=output_dir)
        self.output['model'] = self.output['model'].format(output_dir=output_dir)
        self.output['eval_results'] = self.output['eval_results'].format(output_dir=output_dir)
        self.output['test_results'] = self.output['test_results'].format(output_dir=output_dir)

        self.__score_model(classifier, x_eval, y_eval, 'eval')
        self.__score_model(classifier, x_test, y_test, 'test')

        persistence.obj_to_bin(vectorizer, self.output['vectorizer'])
        persistence.obj_to_bin(classifier, self.output['model'])

        print(f'Training complete, check {output_dir} for results.')

    def __score_model(self, classifier, features, labels, title):
        classes = persistence.json_to_obj(self.input['label_dict'])
        plot_confusion_matrix(
            estimator=classifier,
            X=features,
            y_true=labels,
            display_labels=classes.values(),
            normalize='true')
        plt.title(title)
        plt.savefig(self.output['{title}_results'.format(title=title)])

    def __get_timestamp(self):
        return datetime.datetime.today().strftime('%Y%m%dT%H%M%S')

'''Defines a pipeline step which makes a prediction with the classification model.

'''

from mccore import persistence
from mccore.classifier import Classifier

from src.step import Step

class PredictClassifier(Step):
    '''Defines a pipeline step which makes a prediction with the classification model.

    '''

    def __init__(self):
        super(PredictClassifier, self).__init__()
        self.input = {
            'vectorizer': 'models/classifier_vec.pickle',
            'model': 'models/classifier_mdl.pickle',
            'label_dict': 'data/processed/label_dictionary.json'
        }
        self.output = {
        }

    def run(self):
        '''Runs the pipeline step.

        '''
        classifier = Classifier(
            persistence.bin_to_obj(self.input['vectorizer']),
            persistence.bin_to_obj(self.input['model']),
            persistence.json_to_obj(self.input['label_dict'])
        )
        label, confidence = classifier.predict(self.input['--filename'])

        print('Predicted class \'{label}\' with {confidence:.2f}% confidence.'
              .format(label=label, confidence=confidence*100))

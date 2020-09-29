'''Defines a pipeline step which trains the ner model.

'''

import random

from spacy.util import minibatch, compounding

from mccore import ner
from mccore import persistence

from src.step import Step

class TrainNerModel(Step):
    '''Defines a pipeline step which trains the ner model.

    '''

    def __init__(self):
        super(TrainNerModel, self).__init__()
        self.input = {
            'train_data': 'data/processed/ner_labelled.pickle',
        }
        self.output = {
            'model': 'models/ner_mdl.pickle',
        }

    def run(self):
        '''Runs the pipeline step.

        '''
        iterations = 10
        train_data = persistence.bin_to_obj(self.input['train_data'])
        nlp, ner_pipe = ner.get_model()

        # Add labels
        for _, annotations in train_data:
            for ent in annotations.get("entities"):
                ner_pipe.add_label(ent[2])

        # get names of other pipes to disable them during training
        other_pipes = [pipe for pipe in nlp.pipe_names if pipe != "ner"]
        with nlp.disable_pipes(*other_pipes):  # only train NER
            nlp.begin_training()
            for _ in range(iterations):
                random.shuffle(train_data)
                losses = {}
                # batch up the examples using spaCy's minibatch
                batches = minibatch(train_data, size=compounding(4.0, 32.0, 1.001))
                for batch in batches:
                    texts, annotations = zip(*batch)
                    nlp.update(texts, annotations, drop=0.5, losses=losses)
                #self.print(losses['ner'])

        persistence.obj_to_bin(nlp.to_bytes(), self.output['model'])

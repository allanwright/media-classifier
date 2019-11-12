#from __future__ import unicode_literals, print_function
import random
import spacy
from spacy.util import minibatch, compounding
from mccore import EntityRecognizer
from mccore import ner
from mccore import persistence

def train():
    '''Trains the named entity recognition model.

    '''
    iterations=10
    train_data = persistence.bin_to_obj('data/processed/ner_labelled.pickle')
    nlp, ner_pipe = ner.get_model()

    # Add labels
    for _, annotations in train_data:
        for ent in annotations.get("entities"):
            ner_pipe.add_label(ent[2])

    # get names of other pipes to disable them during training
    other_pipes = [pipe for pipe in nlp.pipe_names if pipe != "ner"]
    with nlp.disable_pipes(*other_pipes):  # only train NER
        nlp.begin_training()
        for i in range(iterations):
            random.shuffle(train_data)
            losses = {}
            # batch up the examples using spaCy's minibatch
            batches = minibatch(train_data, size=compounding(4.0, 32.0, 1.001))
            for batch in batches:
                texts, annotations = zip(*batch)
                nlp.update(texts, annotations, drop=0.5, losses=losses)
            print("Losses", losses)
    
    persistence.obj_to_bin(nlp.to_bytes(), 'models/ner_mdl.pickle')

def predict(filename):
    ''' Makes a prediction using the named entity recognition model.

    Args:
        input (filename): The filename to evaluate.
    '''
    nlp, _ = ner.get_model()
    nlp_bytes = persistence.bin_to_obj('models/ner_mdl.pickle')
    nlp.from_bytes(nlp_bytes)
    recognizer = EntityRecognizer(nlp)
    print(recognizer.predict(filename))
from __future__ import unicode_literals, print_function
import plac
import random
from pathlib import Path
import spacy
from spacy.util import minibatch, compounding
from mccore import persistence
from mccore import preprocessing

def train():
    '''Trains the named entity recognition model.

    '''
    iterations=10

    # Load training data
    train_data = persistence.bin_to_obj('data/processed/ner_labelled.pickle')

    nlp = spacy.blank('en')
    ner = nlp.create_pipe('ner')
    nlp.add_pipe(ner, last=True)

    # Add labels
    for _, annotations in train_data:
        for ent in annotations.get("entities"):
            ner.add_label(ent[2])

    # get names of other pipes to disable them during training
    other_pipes = [pipe for pipe in nlp.pipe_names if pipe != "ner"]
    with nlp.disable_pipes(*other_pipes):  # only train NER
        nlp.begin_training()
        for itn in range(iterations):
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
    filename = preprocessing.prepare_input(filename)
    nlp = spacy.blank('en')
    ner = nlp.create_pipe('ner')
    nlp.add_pipe(ner, last=True)
    nlp_bytes = persistence.bin_to_obj('models/ner_mdl.pickle')
    nlp.from_bytes(nlp_bytes)
    doc = nlp(filename)
    print("Entities", [(ent.text, ent.label_) for ent in doc.ents])
    print("Tokens", [(t.text, t.ent_type_, t.ent_iob) for t in doc])
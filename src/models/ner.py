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
    model = None
    new_model_name='new_model'
    output_dir='models/'
    n_iter=10

    LABEL = ['title', 'source', 'season_id', 'episode_id',
            'resolution', 'episode_name', 'extension',
            'encoding', 'year']
    
    TRAIN_DATA = persistence.bin_to_obj('data/processed/ner_labelled.pickle')

    """ with open ('data/processes/spacy.pickle', 'rb') as fp:
        TRAIN_DATA = pickle.load(fp) """

    nlp = spacy.blank('en')
    ner = nlp.create_pipe('ner')
    nlp.add_pipe(ner, last=True)

    """ for i in LABEL:
        ner.add_label(i) """

    for _, annotations in TRAIN_DATA:
        for ent in annotations.get("entities"):
            ner.add_label(ent[2])

    optimizer = nlp.begin_training()

    # get names of other pipes to disable them during training
    other_pipes = [pipe for pipe in nlp.pipe_names if pipe != "ner"]
    with nlp.disable_pipes(*other_pipes):  # only train NER
        # reset and initialize the weights randomly – but only if we're
        # training a new model
        if model is None:
            nlp.begin_training()
        for itn in range(n_iter):
            random.shuffle(TRAIN_DATA)
            losses = {}
            # batch up the examples using spaCy's minibatch
            batches = minibatch(TRAIN_DATA, size=compounding(4.0, 32.0, 1.001))
            for batch in batches:
                texts, annotations = zip(*batch)
                nlp.update(
                    texts,  # batch of texts
                    annotations,  # batch of annotations
                    drop=0.5,  # dropout - make it harder to memorise data
                    losses=losses,
                )
            print("Losses", losses)

    # test the trained model
    """ for text, _ in TRAIN_DATA:
        doc = nlp(text)
        print("Entities", [(ent.text, ent.label_) for ent in doc.ents])
        print("Tokens", [(t.text, t.ent_type_, t.ent_iob) for t in doc]) """

    for s in ['dark s01 e01 mp4', 'back to the future i mkv']:
        doc = nlp(s)
        print("Entities", [(ent.text, ent.label_) for ent in doc.ents])
        print("Tokens", [(t.text, t.ent_type_, t.ent_iob) for t in doc])

    # save model to output directory
    if output_dir is not None:
        output_dir = Path(output_dir)
        if not output_dir.exists():
            output_dir.mkdir()
        nlp.to_disk(output_dir)
        print("Saved model to", output_dir)

        # test the saved model
        print("Loading from", output_dir)
        nlp2 = spacy.load(output_dir)
        """ for text, _ in TRAIN_DATA:
            doc = nlp2(text)
            print("Entities", [(ent.text, ent.label_) for ent in doc.ents])
            print("Tokens", [(t.text, t.ent_type_, t.ent_iob) for t in doc]) """

def predict(filename):
    ''' Makes a prediction using the named entity recognition model.

    Args:
        input (filename): The filename to evaluate.
    '''
    filename = preprocessing.prepare_input(filename)
    #x_train, _, _, _ = datasets.get_train_test_data()
    #df = pd.DataFrame()
    #df['name'] = x_train
    #df['entity'] = ''
    nlp = spacy.load(Path('models/'))
    doc = nlp(filename)
    print("Entities", [(ent.text, ent.label_) for ent in doc.ents])
    print("Tokens", [(t.text, t.ent_type_, t.ent_iob) for t in doc])
    """ for i in range(x_train.shape[0]):
        s = x_train[i]
        doc = nlp(s)
        df.loc[i, 'entity'] """
#%%
from __future__ import unicode_literals, print_function

import json
import logging
import sys
import numpy as np
import pandas as pd

import os
import pickle

import random
from pathlib import Path
import spacy
from spacy.util import minibatch, compounding
from src import datasets

def tsv_to_json_format(input_path,output_path,unknown_label):
    try:
        f=open(input_path,'r') # input file
        fp=open(output_path, 'w') # output file
        data_dict={}
        annotations =[]
        label_dict={}
        s=''
        start=0
        for line in f:
            word,entity=line.split('\t')
            s+=word+" "
            entity=entity[:len(entity)-1]
            if entity!=unknown_label:
                if len(entity) != 1:
                    d={}
                    d['text']=word
                    d['start']=start
                    d['end']=start+len(word)-1  
                    try:
                        label_dict[entity].append(d)
                    except:
                        label_dict[entity]=[]
                        label_dict[entity].append(d) 
            start+=len(word)+1
            if entity == 'extension':
                data_dict['content']=s
                s=''
                label_list=[]
                for ents in list(label_dict.keys()):
                    for i in range(len(label_dict[ents])):
                        if(label_dict[ents][i]['text']!=''):
                            l=[ents,label_dict[ents][i]]
                            for j in range(i+1,len(label_dict[ents])): 
                                if(label_dict[ents][i]['text']==label_dict[ents][j]['text']):  
                                    di={}
                                    di['start']=label_dict[ents][j]['start']
                                    di['end']=label_dict[ents][j]['end']
                                    di['text']=label_dict[ents][i]['text']
                                    l.append(di)
                                    label_dict[ents][j]['text']=''
                            label_list.append(l)                          
                            
                for entities in label_list:
                    label={}
                    label['label']=[entities[0]]
                    label['points']=entities[1:]
                    annotations.append(label)
                data_dict['annotation']=annotations
                annotations=[]
                json.dump(data_dict, fp)
                fp.write('\n')
                data_dict={}
                start=0
                label_dict={}
    except Exception as e:
        logging.exception("Unable to process file" + "\n" + "error = " + str(e))
        return None

def write_spacy_file(input_file=None, output_file=None):
    try:
        training_data = []
        lines=[]
        with open(input_file, 'r') as f:
            lines = f.readlines()

        for line in lines:
            data = json.loads(line)
            text = data['content']
            entities = []
            for annotation in data['annotation']:
                point = annotation['points'][0]
                labels = annotation['label']
                if not isinstance(labels, list):
                    labels = [labels]

                for label in labels:
                    entities.append((point['start'], point['end'] + 1 ,label))


            training_data.append((text, {"entities" : entities}))

        #print(training_data)

        with open(output_file, 'wb') as fp:
            pickle.dump(training_data, fp)

    except Exception as e:
        logging.exception("Unable to process " + input_file + "\n" + "error = " + str(e))
        return None

df = pd.read_csv('data/interim/ner_labelled.csv')
df = df[['word', 'entity']]
df.to_csv(
    'data/interim/ner_labelled.tsv',
    sep='\t',
    header=False,
    index=False)

#%%
tsv_to_json_format(
    "data/interim/ner_labelled.tsv",
    'data/interim/ner_labelled.json',
    'na')

write_spacy_file(
    'data/interim/ner_labelled.json',
    'data/interim/spacy.pickle')

model = None
new_model_name='new_model'
output_dir='models/'
n_iter=10

LABEL = ['title', 'source', 'season_id', 'episode_id',
         'resolution', 'episode_name', 'extension',
         'encoding', 'year']

with open ('data/interim/spacy.pickle', 'rb') as fp:
    TRAIN_DATA = pickle.load(fp)

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

#%%
""" def predict():
    x_train, _, _, _ = datasets.get_train_test_data()
    df = pd.DataFrame()
    df['name'] = x_train
    df['entity'] = ''
    nlp = spacy.load(Path('models/'))
    for i in range(x_train.shape[0]):
        s = x_train[i]
        doc = nlp(s)
        df.loc[i, 'entity']
    print(df)

predict() """
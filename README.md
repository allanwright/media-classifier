## Introduction
Media-Classifier is a project designed to classify and extract named entities from filenames. The intended purpose of this project is to create machine learning models that help to automate the process of organising media collections.

It was originally built to classify filenames as either movies, tv shows, music, apps or other files but I've since reduced the number of classes to focus on what really matters to me. The classifier now identifies filenames as either a movie, tv show or other file.

## Development Environment
Development is performed inside a docker container, using the Visual Studio Code Remote - Container extension. Just open the project folder in the dev container and the environment will be fully setup automatically. Please note that the training and test data is not included with this repository.

## Command Line Interface
Media-Classifier has a CLI for executing everything required to train and validate the machine learning models.

```shell
Usage:
    Usage:
        mc aquire --source <source>
        mc predict --model <model> --filename <filename> [--id <id>]
        mc process [--model <model>]
        mc promote --model <model> --id <model>
        mc train --model <model>

    Arguments:
        -s <source>, --source <source>          Data source (kraken, pig, test, xerus, yak)
        -m <model>, --model <model>             Type of model (classifier, ner)
        -i <id>, --id <id>                      The model id [default: latest]
        -f <filename>, --filename <filename>    The filename to evaluate

    Pipelines:
        aquire                  Aquires training or test data
        predict                 Uses a model to make a prediction
        process                 Processes training and test data
        promote                 Promotes a model to production
        train                   Trains a model
```

## Classifier
The classifier is a multi class logistic regression model. I did expirment with a CNN with attention, but logistc regression proved to be just as effective, and required much less effort to implement.

The raw training data currently weighs in at around 700k samples but a lot of unusable junk quickly cuts that down to around 450k samples. The training data is then augmented and classes balanced by undersampling over represented classes. An 80/20 split is used for training and validation data, leaving a total of 90k samples for training and validation. More data is aquired as model drift occurs and retraining is required.

## Named Entity Recognition
The named entity recognition model is trained using Spacy on 500 samples. The reason for this is that unlike the classification data, all of the named entity recognition data needs to be human labelled, or at least human validated. After training the initial version of the model on 250 samples, I then used the model to label more training data, followed by manual verification.

The named entity recognition model also uses a custom tokenizer to ensure tokens are always complete words. Two sets of processing occur when performing named entity recognition prediction, one which processes the input passed to the model and the other which processes the raw input for mapping to the entity values in order to retain correct casing and punctuation.
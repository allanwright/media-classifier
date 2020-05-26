## Introduction
Media-Classifier is a project designed to classify and extract named entities from filenames.

## Environment Setup
Development is performed inside a docker container using the Visual Studio Code Remote - Container extension. Just open the project folder in the dev container and the environment will be fully setup automatically. Please note however that the training and test data is not included with this repository.

## Command Line Interface
Media-Classifier has a CLI for executing all required steps in training the text classification and named entity recognition models.

```shell
Usage:
    mc run <pipeline>
    mc predict <model> <filename>

Arguments:
    <pipeline>          The name of the pipeline to run
    <model>             Model to use for predictions (classifier, ner)
    <filename>          The filename to evaluate

Pipelines:
    aquire-data         Aquires training and test data
    process-data        Processes training and test data
    train-classifier    Trains the classification model
    train-ner           Trains the ner model
    all-the-things      Does all the things
```
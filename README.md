## Introduction
Media-Classifier is a project designed to classify and extract named entities from filenames.

## Environment Setup
Development is performed inside a docker container using the Visual Studio Code Remote - Container extension. Just open the project folder in the dev container and the environment will be fully setup automatically. Please note however that the training and test data is not included with this repository/

## Command Line Interface
Media-Classifier has a CLI for executing all required steps in training the text classification and named entity recognition models.

```shell
Usage:
    mc aquire <source>
    mc process <step>
    mc train <model>
    mc predict <model> <filename>
    
Arguments:
    <source>    Source to aquire data from (pig, kraken, xerus, yak, prediction)
    <step>      Processing step to run (all, merge, feature)
    <model>     Model to train/predict (baseline, cnn, ner)
    <filename>  The filename to evaluate
```
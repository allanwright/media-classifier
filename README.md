## Introduction
Media-Classifier is a text classification project designed to classify media based on filename.

## Environment Setup
Development is performed inside a docker container using the Visual Studio Code Remote - Container extension.  Just open the project folder in the dev container and the environment will be fully setup automatically.

## Command Line Interface
Media-Classifier has a CLI for executing all required steps in training the text classification model.

```shell
Usage:
    mc aquire <source>  Downloads raw data from the specified source.
    mc process          Processes the raw data.
    mc train            Trains the model.
    mc eval             Evaluates the model.

Arguments:
    <source>    Source to aquire data from (pig, kraken, xerus, yak)
```
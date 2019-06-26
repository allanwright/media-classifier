## Introduction
Media-Classifier is a text classification project designed to classify media based on filename.

## Environment Setup
Development is performed inside a docker container using the Visual Studio Code Remote - Container extension.  Once the dev container has been built, follow these steps to complete the setup (this will hopefully be automated soon):

```shell
pip install --upgrade pip && \
conda env update --file workspace/environment.yml && \
python setup.py install
```
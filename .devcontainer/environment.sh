#!/bin/bash

# In a Dockerfile:
# RUN apt-get update
# executes inside the container:
# sh -c 'apt-get update'
#
# .bashrc includes the line
# source "/cntk/activate-cntk"
# .bashrc is only loaded for interactive bash shells
#
# /cntk/activate-cntk includes conda in the path (among other things)
# /cntk/activate-cntk includes logic to only work with bash

source "/cntk/activate-cntk"

pip install --upgrade pip && \
conda env update --file /workspace/environment.yml && \
python setup.py install

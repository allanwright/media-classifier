#!/bin/bash

source "/cntk/activate-cntk"

pip install --upgrade pip && \
conda env update --file /workspace/environment.yml
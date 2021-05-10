#!/usr/bin/env bash

VENVNAME=venv_ass2

python3 -m venv $VENVNAME
source $VENVNAME/bin/activate
pip install --upgrade pip

pip install ipython

test -f requirements.txt && pip install -r requirements.txt

deactivate
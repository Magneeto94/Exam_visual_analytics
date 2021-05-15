#!/usr/bin/env bash

#Environment name
VENVNAME=ass4_venv

#Activate environment
source $VENVNAME/bin/activate

#Upgrade pip
pip install --upgrade pip

# problems when installing from requirements.txt
test -f requirements.txt && pip install -r requirements.txt

#navigate to src folder
cd src

#run script. ## The bashscript runs the python-script and it is possible to give it arguments in the CLI.
# $@ makes sure the script can be givin arguments.
python3 lr_minist.py $@

#deactivate environment
deactivate

## Assingment 4
This project contains 2 .py scripts in the src folder. 2 bash scripts in the Assignment4 folder. You only need to use the bash script to run the python scripts.


### To run the scripts
- Download the repository to your own worker02
- navigate to the Assignment4 folder by using cd.
- Now you should only run the bash script included in the folder, as this automatically will activate the venv, install the requirements.txt file, take optional arguments and then deactivate it again.

To run:
- Run the code of lr_minist.py by running: __bash run_lrminist.sh__
    - You can include arguments, if you don't the script will run the default values 0.8 and 0.2.
- Similarly run the code of nn-minist by running __bash run_nn-minist.sh__ 
    - This script has 4 arguments, which you can choose whether to give the script or not. The default will be 32 in hidden layer 1, 0 in hidden layer 2 and 3, and 100 epochs.
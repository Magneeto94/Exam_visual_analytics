# Classifying posters from genres 

__Description:__
This assignment aims to create a convolutional neural network, that is able to decide a film’s genre based on the poster that relates to that film.

<br>

__Run the script:__

- After cloning the repository to worker02 or locally on your machine with: "git clone https://github.com/Magneeto94/Exam_visual_analytics " you can begin to use the scripts.

- Move to the “Final_project” folder
- To create the virtual environment run: “bash create_venv_final_project.sh”
    - The script will install the dependencies from the requirements.txt file.
    - This will create the virtual environment “venv_FP”
- To run the python scripts, move to the src folder
- Now you should first run: “python data_cleaning.py” 
    - You will not be able to run “CNN_Movie_Posters.py” or “pre_trained_model.py” without having run the “python data_cleaning.py”.
    - The output from this script goes into the data folder in the folder: “Poster_data”
- Now run the “CNN_Movie_Posters.py” script by running: “python CNN_Movie_Posters.py”.
    - This will give three outputs: “classification_report.txt”, “model_architecture.png and “model_performance.png”, to be found in the folder output.
- You can also run the “pretrained_model.py” by running “python pretrained_model.py”
    - Similarly this will give three outputs: “classification_report.txt”, “model_architecture.png and “model_performance.png”, to be found in the folder output.


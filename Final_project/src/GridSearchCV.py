'''
-------------- Import libraries:---------------
'''
# Libraries
import pandas as pd
import os
import wget
import numpy as np
import cv2
from pathlib import Path
import matplotlib.pyplot as plt

# sklearn tools
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import classification_report

# tf tools
import tensorflow
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (Conv2D, 
                                     MaxPooling2D, 
                                     Activation, 
                                     Flatten, 
                                     Dense)
from tensorflow.keras.utils import plot_model
from tensorflow.keras.optimizers import SGD
from tensorflow.keras import backend as K  
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier

'''
-------------- Main function:---------------
'''
def main():
    
    '''
    -------------- Load and prepare data:---------------
    '''
    
    '''we use exactly the same setup as in our CNN script. 
    The only difference is that the model is defined as a function where you can apply Sklearns GridSearchCV'''
    
    
    #Load df
    one_Genre_df = pd.read_csv("output/sorted_genre_df.csv")

    #Define the image path
    image_path = os.path.join("data/Sorted_Posters")

    #Create empty list, where the arrays will be storred
    np_images = []

    #Convert every image in the image_path to numpy arrays
    for image in Path(image_path).glob("*"):
        image_open = cv2.imread(str(image))
        np_images.append(image_open)

    #Convert to numpy array
    np_images=np.array(np_images)

    #Flatten the data and define X and y
    n_samples = len(np_images)
    X = np_images.reshape((n_samples, -1))
    y = one_Genre_df["Genre"]

    X_train, X_test, y_train, y_test = train_test_split(np_images, 
                                                        one_Genre_df["Genre"], 
                                                        train_size = 4051,
                                                        test_size = 1013) 
    '''
    -------------- Define model:---------------
    '''                                                    

    #Define nn_model
    def nn_model(optimizer='sgd'):
        # create a sequential model
        model = Sequential()
        # add input layer of nodes and hidden layer of 32, ReLU activation
        model.add(Conv2D(32,(3, 3), input_shape=(268, 182, 3,), activation="relu"))
        # add input layer of nodes and hidden layer of 16, ReLU activation
        model.add(Conv2D(16, (5, 5), activation="relu"))
        model.add(Flatten())
        # hidden layer of 6 nodes, ReLU activation
        model.add(Dense(6, activation="relu"))
        # classificaiton layer, 6 classes with softmax activation
        model.add(Dense(6, activation="softmax")) 
        # categorical cross-entropy, optimizer defined in function call
        model.compile(loss="categorical_crossentropy", 
                      optimizer=optimizer, 
                      metrics=["accuracy"])

        # return the compiled model
        return model

    '''
    -------------- Estimate the best parameters:---------------
    '''
    #Build the model defined in nn_model
    model = KerasClassifier(build_fn=nn_model, 
                            verbose=1)         # set to 1 for verbose output during training

    # grid search epochs, batch size and optimizer
    optimizers = ['sgd', 'adam', 'lbfgs']
    # range of epochs to run
    epochs = [3, 10]
    # variable batch sizes
    batches = [3, 10]

    # create search grid
    param_grid = dict(optimizer=optimizers, 
                      epochs=epochs, 
                      batch_size=batches)

    grid = GridSearchCV(estimator=model, 
                        param_grid=param_grid, 
                        n_jobs=-1,    # number of CPU cores to use: -1 means use all available
                        cv=5,         # 5-fold cross validation
                        scoring='accuracy')


    grid_result = grid.fit(X_train,y_train)

    # print best results, rounding values to 3 decimal places
    print(f"Best run: {round(grid_result.best_score_,3)} using {grid_result.best_params_}")
    print()

#Define behaviour when called from command line
if __name__ == "__main__":
    main()   
    
    
    
    
    

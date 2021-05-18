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
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import classification_report

# tf tools
import tensorflow
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (Conv2D, 
                                     AveragePooling2D, 
                                     Activation, 
                                     Flatten, 
                                     Dense)
from tensorflow.keras.utils import plot_model
from tensorflow.keras.optimizers import SGD
from tensorflow.keras import backend as K  

'''
-------------- Main function:---------------
'''
def main():
    
    '''
    ---------------------CLEANING DATA:------------------------------
    '''
    #Path to data
    input_file = os.path.join("..", "data", "MovieGenre.csv")

    #Reading data
    data = pd.read_csv(input_file, encoding = "ISO-8859-1")

    #Renaming all movies that have Animation in their title to just being "Animation"
    data["Genre"] = data["Genre"].replace(to_replace=r'^.*Animation(.*)', value='Animation', regex=True)
    data["Genre"] = data["Genre"].replace(to_replace='Drama|Romance', value='Romance', regex=False)
    data["Genre"] = data["Genre"].replace(to_replace=r'^.*Western(.*)', value='Western', regex=True)
    data["Genre"] = data["Genre"].replace(to_replace=r'^.*Sci-Fi(.*)', value='Sci-Fi', regex=True)
    data["Genre"] = data["Genre"].replace(to_replace=r'^.*Horror(.*)', value='Horror', regex=True)

    #Getting all unique cathegories from the dataset:
    unique_data = data.Genre.unique()


    #Iterating through the unique cathegories.
    unique_cathegories = []
    for cat in unique_data:
    
        #We find the Genres, that is only described by cathegory
        # "|" indicate that the movie has more than one genre.
        if "|" not in str(cat):
            unique_cathegories.append(cat)


    # If we want to use the movie title
    # Title
    # Replace the whitespaces in the titles with a underscore
    data["Title"] = data["Title"].str.replace(pat=" ", repl="_")
    
    
    one_Genre_df = data[data.Genre.isin(unique_cathegories)]
    
    
    
    one_Genre_df = one_Genre_df[data.Genre.isin(["Horror", "Western", "Animation"])]
    one_Genre_df = one_Genre_df.reset_index()
    
    
    
    #Iterating through the unique cathegories.
    cathegories = []
    for cat in one_Genre_df["Genre"].unique():
    
        #We find the Genres, that is only described by cathegory
        # "|" indicate that the movie has more than one genre.
        if "|" not in str(cat):
            cathegories.append(cat)

    print(f"The cathegories in the data set: {cathegories}")
    
    
    
    '''
    ------------------------DOWNLOADING IMAGES:----------------------
    '''
    
    try:
        os.mkdir(os.path.join("..", "data", "Poster_data"))
        print("Poster_data was created!")
    except FileExistsError:
        print("Poster_data already exists!")
    
    
    errors = []
    for i in range(len(one_Genre_df)):
    
        index = str(i)
        #Creating name of poster files
        filename = "../data/Poster_data/"+ str(one_Genre_df["Title"][i]) + ".jpg"
        print(filename)
    
        #Accessing the links for the posters
        image_url = one_Genre_df["Poster"][i]
        #print(image_url)
    
        #Error handling.
        #If the poster does not exist: pass, and move on to the next file.
        try:
            image_filename = wget.download(image_url, filename)
        except:
            print("There was an error")
            errors.append(int(index))
            pass
    
    
    sorted_df = one_Genre_df.drop(labels=errors, axis=0).reset_index(drop=True)
    
    #Write to csv-file
    sorted_df.to_csv(os.path.join("..", "data", "sorted_df.csv"))
    
    
if __name__ == "__main__":
    main() 
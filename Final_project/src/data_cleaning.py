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
    
    #Same procedure as above with different data
    data["Genre"] = data["Genre"].replace(to_replace='Drama|Romance', value='Romance', regex=False)
    data["Genre"] = data["Genre"].replace(to_replace=r'^.*Western(.*)', value='Western', regex=True)
    data["Genre"] = data["Genre"].replace(to_replace=r'^.*Sci-Fi(.*)', value='Sci-Fi', regex=True)
    data["Genre"] = data["Genre"].replace(to_replace=r'^.*Horror(.*)', value='Horror', regex=True)

    #Getting all unique Genres/cathegories from the dataset:
    unique_data = data.Genre.unique()


    #Iterating through the unique cathegories.
    unique_cathegories = []
    for cat in unique_data:
    
        #We find the Genres, that is only described by one cathegory
        #an "|" indicates that the movie has more than one genre.
        if "|" not in str(cat):
            #if the genre/cathegory does not have an "|" in the string it is appended into unique_cathegories.
            unique_cathegories.append(cat)


    # Replacing the whitespaces in the titles with an underscore, to easier work with the data when it is in folders.
    data["Title"] = data["Title"].str.replace(pat=" ", repl="_")
    
    
    one_Genre_df = data[data.Genre.isin(unique_cathegories)]
    
    

    #Chosing which genre we want in the dataset
    one_Genre_df = one_Genre_df[data.Genre.isin(["Sci-Fi", "Documentary", "Animation", "Romance"])]
    #Resetting index
    one_Genre_df = one_Genre_df.reset_index()
    
    
    
    #Iterating through the unique cathegories.
    cathegories = []
    for cat in one_Genre_df["Genre"].unique():
    
        #We find the Genres, that is only described by cathegory
        # "|" indicate that the movie has more than one genre.
        if "|" not in str(cat):
            cathegories.append(cat)

    print(f"\n The cathegories in the data set: {cathegories}")
    
    
    
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
    
    #Dropping data where no poster was found and resetting index of the dataframe
    sorted_df = one_Genre_df.drop(labels=errors, axis=0).reset_index(drop=True)
    
    #Write to csv-file
    sorted_df.to_csv(os.path.join("..", "data", "sorted_df.csv"))
    
    
if __name__ == "__main__":
    main() 
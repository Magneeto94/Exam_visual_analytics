'''
-------------------Importing packages--------------------------

'''

import os
import sys
sys.path.append(os.path.join(".."))
import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path #Importing Path that we are going to acces our files with.
import csv
import pandas as pd
import argparse


'''
-------------------Creating the script--------------------------

'''
def main():
    
    
    ap = argparse.ArgumentParser(description = "[INFO] creating benchmark classifier")
    
    #The script takes the target image as an argument.
    ap.add_argument("-TI", #flag
                    "--target_image", 
                    required=False, # You do not need to give any arguments, but you can.
                    # If no argument is given, the script will be run with image_0006 as the target image.
                    default="image_0006.jpg",
                    type=str,
                    help="Type in the name of the image you would like to compare with.")
    
    
    args = vars(ap.parse_args())
    
    #Putting the argument into a variable and making sure it is a string
    input_image_name = str(args["target_image"])
    

    
    
    
    
    #writing the data path to where the flower pictures are.
    data_path = os.path.join("..", "data", "17flowers", "jpg")

    #writing the path to where the csv-file endproduckt
    outpath = os.path.join("..", "output", "temporary.csv")


    #Creating containers for the picture data.
    results = []
    new_results = ["Filename", "Distance"]

    
    
    
    # The path to the target image, based on the input of the user.
    target_image = cv2.imread(os.path.join("..", "data", "17flowers", "jpg", input_image_name))

    
    
    #Creating a histogram from the picture
    target_image_hist = cv2.calcHist([target_image], [0,1,2], None, [8,8,8], [0,256, 0,256, 0,256])

    #Normalizing the picture
    normalized_target_hist = cv2.normalize(target_image_hist, target_image_hist, 0,255, cv2.NORM_MINMAX)


    #Creating a for-loop that can run through the folder with all the flower pictures.
    for image in Path(data_path).glob("*.jpg"):
    
        #Reading each image
        image_open = cv2.imread(str(image))
    
        #Making the pictures into histograms
        extract_hist = cv2.calcHist([image_open], [0,1,2], None, [8,8,8], [0,256, 0,256, 0,256])
    
        #Normalizing the histograms.
        normalize_hist = cv2.normalize(extract_hist, extract_hist, 0,255, cv2.NORM_MINMAX)
    
        #Calculating the result of the distance between every image and the target image.
        result = (round(cv2.compareHist(normalized_target_hist, normalize_hist, cv2.HISTCMP_CHISQR), 2))
    
        #Using an if-statement to make sure identical pictures is not compared.
        if result > 0: #If the result is 0, the pictures to be compared are identical.
            results.append(f"{image}, {result}") #appending the result and the names for the compared picture to the results list.
    
    #Sorting the results so they are in alphabetical order. That way picture 0001 apear first then pic 0002, 0003 ect.
    sorted_results = sorted(results)



    '''
    -------------------Now writing to a csv-file.--------------------------
    '''

    #before creating the for-loop.
    #writing a csv-file where the top will be Filename and Distance
    with open(outpath, "w", encoding="utf-8") as distance_file:
        writer = csv.writer(distance_file)    
        writer.writerow(["Filename", "Distance"])
    

    #Now creating a for-loop that runs through the sorted results.
    for result in sorted_results:
    
        #Slicing the name down to only containing, ex. "image_0001.jpg", instead of the full path.
        sliced_names = result[22:36]
    
        #Slicing the result from the rest so only the difference is left.
        sliced_result = result[38:len(result)]
    
        '''
        Now writing our sliced names and sliced results to the csv-file deffined in the outpath and appending each image and its distance.
        This time 'a' is used instead of 'w' because we are appending and don't want to overwrite the Filename and Distance written in 
        earlier.
        '''
        with open(outpath, 'a', newline='') as distance_file:
            writer = csv.writer(distance_file)
            writer.writerow([sliced_names, sliced_result])
    
    
        #Also appending to the list new_results above
        new_results.append(result[22:len(result)])
    
    #Reading the csv I have just created as a dataframe
    distance_df = pd.read_csv("../output/temporary.csv")
    
    #Sorting the data acording to distance
    distance_df = distance_df.sort_values(by=['Distance'])
    
    distance_df = distance_df.reset_index(drop=True)
    
    #Saving the sorted dataframe as a csv.
    distance_df.to_csv(os.path.join("..", "output", "distance.csv"), sep=',')
    
    print(f"Printing the 10 most similar histograms to image {input_image_name}")
    print(distance_df.head(10))

          
if __name__ =='__main__':
    main()
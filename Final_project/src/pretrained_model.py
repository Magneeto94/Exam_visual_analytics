# tf tools
import tensorflow as tf
from tensorflow.keras.utils import plot_model

# image processsing
from tensorflow.keras.preprocessing.image import (load_img,
                                                  img_to_array,
                                                  ImageDataGenerator)
# VGG16 model
from tensorflow.keras.applications.vgg16 import (preprocess_input,
                                                 decode_predictions,
                                                 VGG16)

# layers
from tensorflow.keras.layers import (Flatten, 
                                     Dense, 
                                     Dropout)
# generic model object
from tensorflow.keras.models import Model

# optimizers
from tensorflow.keras.optimizers import SGD, Adam

#scikit-learn
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report


# for plotting
import numpy as np
import matplotlib.pyplot as plt
import os, sys
import pandas as pd
from pathlib import Path
import cv2

def plot_history(H, epochs):
    # Visualize performance
    plt.style.use("fivethirtyeight")
    fig = plt.figure()
    plt.plot(np.arange(0, epochs), H.history["loss"], label="train_loss")
    plt.plot(np.arange(0, epochs), H.history["val_loss"], label="val_loss")
    plt.plot(np.arange(0, epochs), H.history["accuracy"], label="train_acc")
    plt.plot(np.arange(0, epochs), H.history["val_accuracy"], label="val_acc")
    plt.title("Training Loss and Accuracy")
    plt.xlabel("Epoch #")
    plt.ylabel("Loss/Accuracy")
    plt.legend()
    plt.tight_layout()
    plt.show()
    #Save the history as model_performance.png
    fig.savefig("../output/pretrained_model_performance.png")
    
    
    
def main():

    #Define path to data frame created in the data_cleaning.py script.
    df_path = os.path.join("..", "data", "sorted_df.csv")
    
    #Reading the dataframe
    sorted_df = pd.read_csv(df_path)
    
    
    #Iterating through the unique cathegories, to create a list genres
    cathegories = []
    for cat in sorted_df["Genre"].unique():
    
        #We find the Genres, that is only described by cathegory
        # "|" indicate that the movie has more than one genre.
        if "|" not in str(cat):
            cathegories.append(cat)

    print(f"The cathegories in the data set: {cathegories}")
    
    
    
    #Define the image path
    image_path = os.path.join("..", "data", "Poster_data")

    #Create empty dataframe, where the arrays will be storred. And the made into a list later again.
    np_images = pd.DataFrame(columns=["Title", "np_array"])

    #Convert every image in the image_path to numpy arrays
    for image in Path(image_path).glob("*jpg"):
        
        #Resizing the images while we are loading them, to make the model run faster
        img = load_img(image, target_size=(32, 32))
        
        #Converting to np array, so we can work with it in Keras
        image_array = img_to_array(img)
        
        #Saving to datafram
        np_images = np_images.append({'Title' : str(image)[20:len(str(image))-4],
                                      'np_array' : image_array}, 
                                       ignore_index=True)
    
    
    
    #Sorting np_images in alphabetical order
    sorted_np_images_df = np_images.sort_values(by=['Title']).reset_index(drop=True)
    
    
    #Sorting sorted_df in alphabetical order.
    alpha_df = sorted_df.sort_values(by=['Title']).reset_index(drop=True)
    
    
    
    #Merging the 2 dataframes by "Title"
    merged_df = pd.merge(alpha_df, sorted_np_images_df, on="Title")
    
    
    
    #Removing duplicates and resetting index in merged_df.
    merged_df = merged_df.drop_duplicates(subset=['Title']).reset_index(drop=True)
    
    
    
    
    for i in range(len(merged_df["np_array"])):
    
        pre_image = preprocess_input(merged_df["np_array"][i])
    
        merged_df["np_array"][i] = pre_image
        

        
        
    '''
    We are now making the np_array series in the merged_df into a list.
    It is transformed because the CNN model that we train later in the script
    does not run if the X_train and X_test comes from a dataframe
    '''  
    np_data = merged_df["np_array"].to_list()
    
    
    X_train, X_test, y_train, y_test = train_test_split(np_data, 
                                                        merged_df["Genre"],
                                                        test_size= 0.25,
                                                        random_state = 9)
    
    
    X_train = np.array(X_train)
    X_test = np.array(X_test)
    
    X_train = X_train/255.0
    X_test = X_test/255.0
    
    
    
    # integers to one-hot vectors
    lb = LabelBinarizer()
    y_train = lb.fit_transform(y_train)
    y_test = lb.fit_transform(y_test)

    # initialize label names for CIFAR-10 dataset
    labelNames = cathegories
    
    
    # load model without classifier layers
    model = VGG16(include_top=False, 
                  pooling='avg',
                  input_shape=(32, 32, 3))
    
    
    
    # mark loaded layers as not trainable
    for layer in model.layers:
        layer.trainable = False
    
    
    
    
    # add new classifier layers
    flat1 = Flatten()(model.layers[-1].output)
    class1 = Dense(10, 
                   activation='relu')(flat1)
    output = Dense(4, 
                   activation='softmax')(class1)
    
    # define new model
    model = Model(inputs=model.inputs, 
                  outputs=output)
    
    
    #sgd = SGD(lr=0.001, decay=1e-6, momentum=0.9)
    model.compile(optimizer=Adam(lr=0.001),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    
    
    

    # Save model summary
    plot_model(model, to_file = "../output/pretrained_model_architecture.png", show_shapes=True, show_layer_names=True)

    
    H = model.fit(X_train, y_train, 
              validation_data=(X_test, y_test), 
              batch_size=128,
              epochs=10,
              verbose=1)
    
    
    #View the summary
    model.summary()
    
    
    
    plot_history(H, 10)
    
    
    # Print the classification report
    predictions = model.predict(X_test, batch_size=10)
    print(classification_report(y_test.argmax(axis=1),
                                predictions.argmax(axis=1),
                                target_names=cathegories))
    
    
    
    #Writing results to txt file.
    txt_file = open("../output/pretrained_classification_report.txt", "a")
    txt_file.write(classification_report(y_test.argmax(axis=1),
                                         predictions.argmax(axis=1),
                                         target_names=cathegories))
    txt_file.close()

    
    
if __name__ == "__main__":
    main()
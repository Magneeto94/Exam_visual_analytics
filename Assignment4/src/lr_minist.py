import os
import sys
sys.path.append(os.path.join(".."))

# Import teaching utils
import numpy as np
import utils.classifier_utils as clf_util

# Import sklearn metrics
from sklearn import metrics
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

import argparse


'''
-------------------Defining mainfunction-----------------------
'''
def main():
    
    '''
    --------------Defining command line arguments---------------
    '''
    
    ap = argparse.ArgumentParser(description = "[INFO] creating benchmark classifier")
    
    #first argument: size of the training data:
    ap.add_argument("-trs", #flag
                    "--train_size", 
                    required=False, # You do not need to give any arguments, but you can.
                    default=0.8, # If you don't the default is 0.8 or 80%
                    type=float, # The input has has to be a float.
                    help="The size of your training data")
    
    
    ap.add_argument("-tes",
                    "--test_size",
                    required=False,
                    default=0.2,
                    type=float,
                    help="The size of your test data")
    
    args = vars(ap.parse_args())
    
    # Creating variables that can be put in where the commandline arguments are used.
    train_s = args["train_size"]
    test_s = args["test_size"]
    
    
    
    '''
    ----------Creating test and training data-------------------
    '''
    
    # Fetching dataset: mnist.
    X, y = fetch_openml('mnist_784', version=1, return_X_y=True)
    
    # Transforming them to numpy arrays.
    #X = the pictures, y = the category.
    X = np.array(X)
    y = np.array(y)
    
    
    #Creating training and test data.
    X_train, X_test, y_train, y_test = train_test_split(X, 
                                                        y, 
                                                        #random_state=9,
                                                        train_size=train_s, 
                                                        test_size=test_s)
    
    #Normalizing the data by deviding it by 255
    X_train_scaled = X_train/255.0
    X_test_scaled = X_test/255.0
    
    
    
    clf = LogisticRegression(penalty='none', 
                         tol=0.1, 
                         solver='saga',
                         multi_class='multinomial').fit(X_train_scaled, y_train)
    
    
    y_pred = clf.predict(X_test_scaled)
    
    
    #method 2
    accuracy = accuracy_score(y_test, y_pred)
    print(accuracy)
    
    
    cm = metrics.classification_report(y_test, y_pred)
    print(cm)
    
if __name__ =='__main__':
    main()
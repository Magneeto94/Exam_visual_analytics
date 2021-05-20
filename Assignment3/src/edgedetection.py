'''
--------------Import packages:----------------------

'''
import os
import sys
sys.path.append(os.path.join(".."))
import cv2
import numpy as np
#Importing Image from PIL to crop the image an easier way.
from PIL import Image


def main():
    
    '''
    ------------------Reading in image file:-------------------
    '''
    fname = os.path.join("..", "data", "Jefferson_quote.jpg")
    image = cv2.imread(fname)
    
    '''
    --------------------------Drawing rectangular box:-----------------------------
    '''
    #Creating the rectangle, by looking at the image and trying to find coordinates that match.
    
    #Saving the height and width of the image:
    height = image.shape[0] #Index 0 is height of an image.
    width = image.shape[1] #Index 1 is the width of an image.
    
    
    # Setting the points for cropped image. Found by trying different coordinates on the in the pictures.
    left = width//4
    top = height//4
    right = (width//4)*3
    bottom = (height//8)*7
    
    '''
    The following parameters for drawing a rectangle with "cv2.rectangle" is:
                          image,  start point, end point,   color,  line thickness
    '''
    image = cv2.rectangle(image, (left,top), (right,bottom), (0, 255,0), 3)
    
    #saving image to output folder.
    cv2.imwrite(os.path.join("..", "output", "image_with_ROI.jpg"), image)




    '''
    ---------------------------Cropping image:------------------------------------

    '''

    image = cv2.imread(fname)

    #Reading the image again.
    im = Image.open(fname)
  
    # Cropped image of above dimension 
    im1 = im.crop((left, top, right, bottom)) 

    #Saving the cropped image, as it is in the wrong format, but if we reload it with the with cv2, it can be worked with again.
    im1.save(os.path.join("..", "output", "image_cropped.jpg"))



    '''
    ---------------------------------------Edge detection:-------------------------
    '''

    #Reading the cropped file.
    crop_file = os.path.join("..", "output", "image_cropped.jpg")
    crop_image = cv2.imread(crop_file)


    #making the image greyscale
    grey_crop = cv2.cvtColor(crop_image, cv2.COLOR_BGR2GRAY)

    
    ###
    ###
    ###
    '''
    Paramethers for cv2.GaussianBlur()
        image
        kernal: width and height, has to be odd
        standard deviation
    '''
    
    #blurring the picture to find borders
    blurred = cv2.GaussianBlur(grey_crop, (5,5), 0)
    
    
    '''
    Paramethers for "cv2.Canny" is:
           image,
           thresh hold 1: Decides how high the value of intensity gradient can be
           thresh hold 2: Decides how low the value of intensity gradient can be
    '''
    canny = cv2.Canny(blurred, 30, 150) 




    # "_" = dummy variable
    (contours, _) = cv2.findContours(canny.copy(),
                 cv2.RETR_EXTERNAL,#Filtering out inner structures
                 cv2.CHAIN_APPROX_SIMPLE) #Finding contours

    #saving Letters edge detection.
    image_contours = (cv2.drawContours(crop_image.copy(), #draw contours
            contours,                      #our list of contours.
            -1,                            #which contours to draw
            (0,255,0),                     #contours color
            2))                            #contour pixel width
           

    #Saving image
    cv2.imwrite(os.path.join("..", "output", "image_letters.jpg"), image_contours)
    
    
if __name__=="__main__":
    main()
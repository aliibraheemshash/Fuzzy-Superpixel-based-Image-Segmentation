import cv2
import numpy as np



def LAB_Color_Features(img):
    # Reads the image


    # Check if the image was loaded correctly
    if img is None:
        print("Error: Could not read the image.")
        return

    # Converts to LAB color space
    img_lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)

    # Get the dimensions of the image
    rows, cols, _ = img_lab.shape


    # Flatten the LAB image for easier processing
    l_values = img_lab[:, :, 0]#.flatten() #we use flatten function change from 2D to 1D array
    a_values = img_lab[:, :, 1]#.flatten() # REREPRESENT the matrix as array  row by row
    b_values = img_lab[:, :, 2]#.flatten()


 
    return img_lab

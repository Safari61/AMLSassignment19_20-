import numpy as np
import cv2
import pandas as pd
import pickle
import os

# ======================================================================================================================
# when running it  on your own computer, please CHANGE THE PATH TO YOUR OWN PATH!!!!
# ======================================================================================================================
DATADIR = "E:\machine learning\project\dataset_AMLS_19-20\celeba\img"  # define the path of images

dataset = []
data = pd.read_csv('E:\machine learning\project\dataset_AMLS_19-20\celeba\labels.csv')
# when running it  on your own computer, please CHANGE THE PATH TO YOUR OWN PATH!!!!
split = data['\timg_name\tgender\tsmiling'].apply(lambda x: pd.Series(x.split('\t')))  # split into different columns
smiling = split[3]  # get the emotion features
img_array = []
smile = []

for img in os.listdir(DATADIR):
    i = int(img.split('.')[0])  # create an integer type of counter to indicate the index of the array of smile[]
    img_array.append(cv2.imread(os.path.join(DATADIR, img),cv2.IMREAD_GRAYSCALE))
    # read pixel values in gray scale of images, because color is not necessary in task A
    if smiling[i] == '-1':
        smile.append(0)
    if smiling[i] == '1':
        smile.append(1)  # change labels from -1 and +1 to 0 and 1, and save labels to an array


# ======================================================================================================================
# when running it  on your own computer, please CHANGE THE PATH TO YOUR OWN PATH!!!!
# ======================================================================================================================
pickle_out = open("E:\machine learning\project\AMLS_assignment_kit\project_organization_example\AMLS_19-20_SN12345678\Datasets\smile.pickle","wb")
pickle.dump(smile, pickle_out)
pickle_out.close()  # save pixel values into a pickle file

pickle_out = open("E:\machine learning\project\AMLS_assignment_kit\project_organization_example\AMLS_19-20_SN12345678\Datasets\img_array_A.pickle","wb")
pickle.dump(img_array, pickle_out)
pickle_out.close()  # save labels into a pickle file
import pandas as pd
import pickle
import cv2
import os

# ======================================================================================================================
# when running it  on your own computer, please CHANGE THE PATH TO YOUR OWN PATH!!!!
# ======================================================================================================================
DATADIR = "E:\machine learning\project\dataset_AMLS_19-20\cartoon_set\img"  # define the path of images

data = pd.read_csv('E:\machine learning\project\dataset_AMLS_19-20\cartoon_set\labels.csv')
# when running it  on your own computer, please CHANGE THE PATH TO YOUR OWN PATH!!!!
img_B = []
split = data['\teye_color\tface_shape\tfile_name'].apply(lambda x: pd.Series(x.split('\t')))  # split into different columns
eye_color = split[1]  # get the eye color features
color = []

for img in os.listdir(DATADIR):
    i = int(img.split('.')[0])   # create an integer type of counter to indicate the index of the array of eye_color[]
    raw = cv2.imread(os.path.join(DATADIR, img))  # read BGR values of images
    new = cv2.resize(raw, (100, 100))  # resize them, we don't need very high resolution
    img_B.append(new)  # save data to an array

    if eye_color[i] == '0':
        color.append(0)
    if eye_color[i] == '1':
        color.append(1)
    if eye_color[i] == '2':
        color.append(2)
    if eye_color[i] == '3':
        color.append(3)
    if eye_color[i] == '4':
        color.append(4)  # save labels to an array
# ======================================================================================================================
# when running it  on your own computer, please CHANGE THE PATH TO YOUR OWN PATH!!!!
# ======================================================================================================================
pickle_out = open("E:\machine learning\project\AMLS_assignment_kit\project_organization_example\AMLS_19-20_SN12345678\Datasets\img_array_B.pickle", "wb")
pickle.dump(img_B, pickle_out)
pickle_out.close()  # save pixel values into a pickle file

pickle_out = open("E:\machine learning\project\AMLS_assignment_kit\project_organization_example\AMLS_19-20_SN12345678\Datasets\eye_color.pickle", "wb")
pickle.dump(color, pickle_out)
pickle_out.close()  # save labels into a pickle file
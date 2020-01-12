# README

# THE MOST IMPORTANT THINGS:
#   1:
#       Please change the file paths into your own paths, when running the codes!!!
#   2:
#       Please first run the data pre-processing codes in the folder of A1, A2, B1, B2 correspondingly, to generate input data for training the model. The files are too big to upload to Github, so you have to run the codes 
#       and genenrate them first, then run the "main.py". You CAN'T directly run the "main.py"!!!
#   3:
#       If any strange (extremely low accuracy) problem occurs, please re-run it. 
#       Keras module has this strange bug, sometimes it will be struck into local optimization point or you are running out of GPU memory, and the accuracy is trapped into a really low accuracy.
#       When this happens, please run it again


# Visualization of training process:
#   You can load the tensorboard logs generated during the training process to visualize the training process, and the these figures are used in my report.


1. Organization of my project
    there are two parts of each task, 
    the first part is to read data from the given dataset and save them in pickle files. The files of codes in this part is located in the A1, A2, B1, B2 folder correspondingly.
    the second part is to model implementation, in detail, to build the model and train it using data from the first part. The codes of this part are saved altogether in the "main.py".
    
2. The role of each file
    The codes in the folder of A1, A2, B1, B2 are used to pre-process and generate data.
    The codes in "main.py" is used to build and train the models for each task.
    
3. The packages required to run your code:
    tensorflow,
    tensorflow.keras.models,
    tensorflow.keras.layers,
    tensorflow.keras.callbacks,
    tensorflow.keras.utils,
    sklearn.model_selection,
    numpy,
    pickle,
    time,
    os.
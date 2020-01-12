import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
import numpy as np
import pickle
import time
import os

# ======================================================================================================================
# Task A1
model_name = "A1-{}".format(int(time.time()))
tensorboard = TensorBoard(log_dir = os.path.join(model_name, str(time.time())))
# define the log name of the training process, which could be used by the tensorboard later

pickle_in = open(r"E:\machine learning\project\AMLS_assignment_kit\project_organization_example\AMLS_19-20_Youhua.Liu_SN19074557\Datasets\img_array_A.pickle","rb")
X = pickle.load(pickle_in)
pickle_in = open(r"E:\machine learning\project\AMLS_assignment_kit\project_organization_example\AMLS_19-20_Youhua.Liu_SN19074557\Datasets\sex.pickle","rb")
y = pickle.load(pickle_in)  # load data

X = np.array(X)
y = np.array(y)  # make sure the input data's format as numpy array
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33)  # split training set and test set

pixels = X_train.reshape(-1, 218, 178, 1)  # reshape data as 4D, which could be accepted by the model
pixels = pixels/255.0  # scale them into the range of [0,1], which could be accepted by the model

model = Sequential()  # initial a sequential CNN network

model.add(Conv2D(32, (3, 3), input_shape=pixels.shape[1:]))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))  # add a convolution layer and corresponding activation and pooling layer

model.add(Conv2D(32, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))  # add a convolution layer and corresponding activation and pooling layer

model.add(Flatten())  # flatten high dimensional data into 1D

model.add(Dense(32))
model.add(Activation('relu'))  # add a dense layer to the model

model.add(Dense(1))
model.add(Activation('sigmoid'))  # add a dense layer to the model, this is also the output layer
#  because this is a binary classification, the activation function is sigmoid, and the size of FC layer is 1.

model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])  # add the loss layer to the model, define the loss function

A1 = model.fit(pixels, y_train, batch_size=32, epochs=7, validation_split=0.2, callbacks = [tensorboard])
# feed data into the model, and start training.
acc_A1_train = A1.history['accuracy'][6]  # get the accuracy of training set
model.save('A1.model')  # save the model

pixels_test = X_test.reshape(-1, 218, 178, 1)
pixels_test = pixels_test/255.0  # prepare the test set

test_result = model.evaluate(pixels_test, y_test)  # feed test set to the model, and get the test results.
print("test loss and accuracy are:", test_result)  # print the test set results
acc_A1_test = test_result[1]  # get the accuracy of test set

# ======================================================================================================================
# Task A2
model_name = "A2-{}".format(int(time.time()))
tensorboard = TensorBoard(log_dir = os.path.join(model_name, str(time.time())))
# define the log name of the training process, which could be used by the tensorboard later
pickle_in = open("E:\machine learning\project\AMLS_assignment_kit\project_organization_example\AMLS_19-20_Youhua.Liu_SN19074557\Datasets\img_array_A.pickle","rb")
X = pickle.load(pickle_in)
pickle_in = open("E:\machine learning\project\AMLS_assignment_kit\project_organization_example\AMLS_19-20_Youhua.Liu_SN19074557\Datasets\smile.pickle","rb")
y = pickle.load(pickle_in)  # load data

X = np.array(X)
y = np.array(y)  # make sure the input data's format as numpy array

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33)  # split training set and test set

pixels = X_train.reshape(-1, 218, 178, 1)  # reshape data as 4D, which could be accepted by the model
pixels = pixels/255.0  # scale them into the range of [0,1], which could be accepted by the model

model = Sequential()  # initial a sequential CNN network

model.add(Conv2D(32, (3, 3), input_shape=pixels.shape[1:]))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))  # add a convolution layer and corresponding activation and pooling layer

model.add(Conv2D(32, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))  # add a convolution layer and corresponding activation and pooling layer

model.add(Flatten())  # flatten high dimensional data into 1D

model.add(Dense(32))
model.add(Activation('relu'))  # add a dense layer to the model

model.add(Dense(1))
model.add(Activation('sigmoid'))  # add a dense layer to the model, this is also the output layer
#  because this is a binary classification, the activation function is sigmoid, and the size of FC layer is 1.
model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])  # add the loss layer to the model, define the loss function

A2 = model.fit(pixels, y_train, batch_size=32, epochs=4, validation_split=0.2, callbacks = [tensorboard])
# feed data into the model, and start training.
acc_A2_train = A2.history['accuracy'][3]  # get the accuracy of training set
model.save('A2.model')  # save the model

pixels_test = X_test.reshape(-1, 218, 178, 1)
pixels_test = pixels_test/255.0  # prepare the test set

test_result = model.evaluate(pixels_test , y_test)  # feed test set to the model, and get the test results.
print("test loss and accuracy are:", test_result)  # print the test set results
acc_A2_test = test_result[1]  # get the accuracy of test set

# ======================================================================================================================
# Task B1
model_name = "B1-{}".format(int(time.time()))
tensorboard = TensorBoard(log_dir = os.path.join(model_name, str(time.time())))
# define the log name of the training process, which could be used by the tensorboard later
pickle_in = open("E:\machine learning\project\AMLS_assignment_kit\project_organization_example\AMLS_19-20_Youhua.Liu_SN19074557\Datasets\img_array_B.pickle","rb")
X = pickle.load(pickle_in)
pickle_in = open("E:\machine learning\project\AMLS_assignment_kit\project_organization_example\AMLS_19-20_Youhua.Liu_SN19074557\Datasets\eye_color.pickle","rb")
y = pickle.load(pickle_in)  # load data

X = np.array(X)
y = np.array(y)  # make sure the input data's format as numpy array

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33)  # split training set and test set

pixels = X_train.reshape(-1, 100, 100, 3)  # reshape data as 4D, which could be accepted by the model
pixels = pixels/255.0  # scale them into the range of [0,1], which could be accepted by the model
y_train = to_categorical(y_train, 5)  # convert labels from integers to binary class matrices

model = Sequential()  # initial a sequential CNN network

model.add(Conv2D(32, (3, 3), input_shape=pixels.shape[1:]))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))  # add a convolution layer and corresponding activation and pooling layer

model.add(Conv2D(32, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))  # add a convolution layer and corresponding activation and pooling layer

model.add(Flatten())  # flatten high dimensional data into 1D

model.add(Dense(32))
model.add(Activation('relu'))  # add a dense layer to the model

model.add(Dense(5))
model.add(Activation('softmax'))  # add a dense layer to the model, this is also the output layer
#  because this is a multi classification, the activation function is softmax, and the size of FC layer is 5.
model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])  # add the loss layer to the model, define the loss function

B1 = model.fit(pixels, y_train, batch_size=32, epochs=10, validation_split=0.2, callbacks = [tensorboard])
# feed data into the model, and start training.
acc_B1_train = B1.history['accuracy'][9]  # get the accuracy of training set
model.save('B1.model')  # save the model

pixels_test = X_test.reshape(-1, 100, 100, 3)
pixels_test = pixels_test/255.0
y_test = to_categorical(y_test, 5)  # prepare the test set

test_result = model.evaluate(pixels_test, y_test)  # feed test set to the model, and get the test results.
print("test loss and accuracy are:", test_result)  # print the test set results
acc_B1_test = test_result[1]  # get the accuracy of test set

# ======================================================================================================================
# Task B2
model_name = "B2-{}".format(int(time.time()))
tensorboard = TensorBoard(log_dir = os.path.join(model_name, str(time.time())))
# define the log name of the training process, which could be used by the tensorboard later
pickle_in = open("E:\machine learning\project\AMLS_assignment_kit\project_organization_example\AMLS_19-20_Youhua.Liu_SN19074557\Datasets\img_array_B.pickle","rb")
X = pickle.load(pickle_in)
pickle_in = open(r"E:\machine learning\project\AMLS_assignment_kit\project_organization_example\AMLS_19-20_Youhua.Liu_SN19074557\Datasets\face_shape.pickle","rb")
y = pickle.load(pickle_in)  # load data

X = np.array(X)
y = np.array(y)  # make sure the input data's format as numpy array

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33)  # split training set and test set

pixels = X_train.reshape(-1, 100, 100, 3)  # reshape data as 4D, which could be accepted by the model
pixels = pixels/255.0  # scale them into the range of [0,1], which could be accepted by the model
y_train = to_categorical(y_train,5)  # convert labels from integers to binary class matrices

model = Sequential()  # initial a sequential CNN network

model.add(Conv2D(32, (3, 3), input_shape=pixels.shape[1:]))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))  # add a convolution layer and corresponding activation and pooling layer

model.add(Conv2D(32, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))  # add a convolution layer and corresponding activation and pooling layer

model.add(Flatten())  # flatten high dimensional data into 1D

model.add(Dense(32))
model.add(Activation('relu'))  # add a dense layer to the model

model.add(Dense(5))
model.add(Activation('softmax'))  # add a dense layer to the model, this is also the output layer
#  because this is a multi classification, the activation function is softmax, and the size of FC layer is 5.
model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])  # add the loss layer to the model, define the loss function

B2 = model.fit(pixels, y_train, batch_size=32, epochs=10, validation_split=0.2, callbacks = [tensorboard])
# feed data into the model, and start training.
acc_B2_train = B2.history['accuracy'][9]  # get the accuracy of training set
model.save('B2.model')  # save the model

pixels_test = X_test.reshape(-1, 100, 100, 3)
pixels_test = pixels_test/255.0
y_test = to_categorical(y_test,5)  # prepare the test set

test_result = model.evaluate(pixels_test, y_test)  # feed test set to the model, and get the test results.
print("test loss and accuracy are:", test_result)  # print the test set results
acc_B2_test = test_result[1]  # get the accuracy of test set

# ======================================================================================================================
#  Print out your results with following format:
print('TA1:{},{};TA2:{},{};TB1:{},{};TB2:{},{};'.format(acc_A1_train, acc_A1_test,
                                                        acc_A2_train, acc_A2_test,
                                                        acc_B1_train, acc_B1_test,
                                                        acc_B2_train, acc_B2_test))


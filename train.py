import numpy as np
import os
import sys
import matplotlib.pyplot as plt
import pandas as pd
#keras
from keras.models import Sequential, Model
from keras.layers import Convolution1D, MaxPooling1D, Flatten, Dense, Embedding, Activation, BatchNormalization, GlobalAveragePooling1D, Input, merge, ZeroPadding1D,LeakyReLU
from keras.preprocessing import sequence
from keras.optimizers import RMSprop, Adam, SGD
from keras.regularizers import l2
import itertools
#checking if keras use gpu
from keras import backend as K
from keras.utils import np_utils # from keras import utils as np_utils
#K.tensorflow_backend._get_available_gpus()
#custom
from vectorize import X, Y, clean_titles
from data_import import logger
# here are the diferent level you can use
#logger.debug("")
#logger.info("")
#logger.warning("")
#logger.error("")
#logger.critical("")

def split_input(X, Y):
    """split the dataset to create the input data ( train_x, train_y, test_x, test_y )

    the size of the training/test set are define by a chosen pourcentage of the training dataset
    (e.g) 90% will generate a test set of 10% and training set of 90% of the original dataset """
    #(arbitrary)
    training_size_pourcentage = 90
    end_x = int((len(X)/100)*training_size_pourcentage)
    print('assigning train data')
    return X[:end_x], Y[:end_x], X[end_x:], Y[end_x:]

train_x, train_y, test_x, test_y = split_input(X, Y)
print(train_y[0:20])
print(train_x.shape)
print(len(train_y))
print(test_x.shape)
print(len(test_y))

#you need to categorise before converting to one hot otherwise it will get the last value of the dataset as the number of classes
"""Categorise data"""
train_y = pd.cut(train_y, bins=[-1,24,49,74,99,np.inf], labels=[0,1,2,3,4])
test_y = pd.cut(test_y, bins=[-1,25,50,75,100,np.inf], labels=[0,1,2,3,4])
"""Convert to one hot encoding"""
train_y = np_utils.to_categorical(train_y, num_classes=5)
test_y = np_utils.to_categorical(test_y, num_classes=5)

logger.info("train_x before the ANN is of shape:",train_x.shape," with train_y lenght : ", train_y)
logger.info("test_x before the ANN is of shape: ",test_x.shape," with test_y lenght : ", test_y)

#the tree following line allow you to save the train data in a file
#np.save("fooooo.npy", X,allow_pickle=True)
#a = np.load("fooooo.npy")
#print(a)

logger.debug("test_x is of lenght :",test_x.value_counts(), " and first 10 instances are :", print(test_x[0:10]))
logger.debug("test_y is of lenght :",test_y.value_counts(), " and first 10 instances are :", print(test_y[0:10]))
logger.debug("train_x is of lenght :",train_x.value_counts(), " and first 10 instances are :", print(train_x[0:10]))
logger.debug("train_y is of lenght :",train_y.value_counts(), " and first 10 instances are :", print(train_y[0:10]))

#seting the model
print('setting the model')
model = Sequential()

model.add(Convolution1D(8, 2, kernel_regularizer=l2(0.005),input_shape=(110,300)))
model.add(BatchNormalization())
model.add(Activation("tanh"))

model.add(Convolution1D(8, 2, kernel_regularizer=l2(0.001)))
model.add(BatchNormalization())
model.add(Activation("tanh"))

model.add(Convolution1D(8, 2, kernel_regularizer=l2(0.001)))
model.add(BatchNormalization())
model.add(Activation("sigmoid")) #don't use relu before softmax

model.add(MaxPooling1D(17))
model.add(Flatten())

model.add(Dense(1, use_bias=True, kernel_regularizer=l2(0.001)))
model.add(BatchNormalization())
model.add(Dense(5, activation='softmax'))
model.add(Dense(5, activation='softmax'))
#model.load_weights('models/detector.finetuned.h5', by_name=True)
model.load_weights('models/detector.h5', by_name=True)
print(model.summary())
#compile
print('compile')
model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["categorical_accuracy"])
history = model.fit(train_x, train_y, validation_data=(test_x, test_y), batch_size=8, epochs=100, shuffle=True, verbose=1)
# list all data in history

print(history.history.keys())

def post_train_plot(history):
    """ plot the result of the training"""
    # summarize history for accuracy
    plt.plot(history.history['categorical_accuracy'])
    plt.plot(history.history['val_categorical_accuracy'])
    plt.title('model accuracy')
    plt.ylabel('categorical_accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()
    # summarize history for loss
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()

post_train_plot(history)
model.save_weights("models/detector.h5")

#debug
#print('model layer##################')
#print(model.layers)
print('model sumary##################')
print(model.summary())
print(train_x.shape)
#for i in range(0, len(train_x)):
    # make a prediction
ynew = model.predict(train_x)
print(ynew[1000:1025])

# show the inputs and predicted outputs
#try:

clean_titles = [''.join(str(e) for e in item) for item in clean_titles]
clean_titles = [item.replace("'", "") for item in clean_titles]
[print(item) for item in Y[1000:1025]]
print(Y[1:20])
print("X=%s, Predicted=%s" % (Y, ynew))

print("X=%s, Predicted=%s" % (clean_titles, ynew))
#except Exception as e:
#    raise

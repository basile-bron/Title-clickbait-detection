import numpy as np
import os
import sys
#keras
from keras.models import Sequential, Model
from keras.layers import Convolution1D, MaxPooling1D, Flatten, Dense, Embedding, Activation, BatchNormalization, GlobalAveragePooling1D, Input, merge, ZeroPadding1D
from keras.preprocessing import sequence
from keras.optimizers import RMSprop, Adam, SGD
from keras.regularizers import l2
import itertools
#checking if keras use gpu
from keras import backend as K
K.tensorflow_backend._get_available_gpus()
#custom
from vectorize import X, Y, clean_titles

def split_input(X, Y):
    """split the dataset to create the input data ( train_x, train_y, test_x, test_y )

    the size of the training/test set are define by a chosen pourcentage of the training dataset
    (e.g) 90% will generate a test set of 10% and training set of 90% of the original dataset """
    #(arbitrary)
    training_size_pourcentage = 90
    end_x = int((len(X)/100)*training_size_pourcentage)
    print('assigning train data')
    return X[:end_x], Y[:end_x], X[end_x:], Y[end_x:]
print(len(Y),"looooooooool")
train_x, train_y, test_x, test_y = split_input(X, Y)
print(train_x.shape)
print(len(train_y))
print(test_x.shape)
print(len(test_y))
#seting the model
print('setting the model')
model = Sequential()

model.add(Convolution1D(8, 2, W_regularizer=l2(0.005),input_shape=(110,300)))
model.add(BatchNormalization())
model.add(Activation("relu"))

model.add(Convolution1D(8, 2, W_regularizer=l2(0.001)))
model.add(BatchNormalization())
model.add(Activation("relu"))

model.add(Convolution1D(8, 2, W_regularizer=l2(0.001)))
model.add(BatchNormalization())
model.add(Activation("relu"))

model.add(MaxPooling1D(17))
model.add(Flatten())

model.add(Dense(1, bias=True, W_regularizer=l2(0.001)))
model.add(BatchNormalization())
model.add(Activation("relu"))

model.load_weights('models/detector.finetuned.h5')

#compile
print('compile')
model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["acc"])
model.fit(train_x, train_y, validation_data=(test_x, test_y), batch_size=8, nb_epoch=20, shuffle=True, verbose=2)

model.save_weights("models/detector.finetuned.h5")

#debug
#print('model layer##################')
#print(model.layers)
print('model sumary##################')
print(model.summary())
print(train_x.shape)
#for i in range(0, len(train_x)):
    # make a prediction
ynew = model.predict(train_x)
# show the inputs and predicted outputs
print("X=%s, Predicted=%s" % (clean_titles, ynew))

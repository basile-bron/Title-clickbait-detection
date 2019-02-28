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
print('import custom')
from vectorize import videos

#SETTING TRAIN DATASET
def setting_input(videos):
    """ """
    x = []
    y = []
    for video in videos:
        print(video.title.shape)
        video.title.extend([video.number_of_capital_letter, video.number_of_exclamation_point, video.number_of_interogation_point])
        x.append(video.title)
        y.append(video.ratings)
        #print(np.array(x).shape)
    print(np.array(x[0]).shape)
    print(len(y))
    return x, y
def split_input(videos):
    """split the dataset to create the input data ( train_x, train_y, test_x, test_y ) """
    #(arbitrary)
    training_size_pourcentage = 90
    end_x = int((len(videos)/100)*training_size_pourcentage)

    print('assigning train data')
    print('x')
    train = [videos[i] for i in range(0, end_x)]
    train_x, train_y = setting_input(train)

    print('y')
    test = [videos[i] for i in range(end_x,len(videos))]
    test_x, test_y = setting_input(train)
    return train_x, train_y, test_x, test_y

train_x, train_y, test_x, test_y = split_input(videos)

#seting the model
print('setting the model')

model = Sequential()

model.add(Convolution1D(8, 2, W_regularizer=l2(0.005),input_shape=(4,110)))
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

#compile
print('compile')
model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["acc"])
model.fit(train_x, train_y, validation_data=(test_x, test_y), batch_size=8, nb_epoch=50, shuffle=True, verbose=2)

#debug
print('model layer##################')
print(model.layers)
print('model sumary##################')
print(model.summary())

import numpy as np
import os
import sys
#keras
from keras.models import Sequential, Model
from keras.layers import Convolution1D, MaxPooling1D, Flatten, Dense, Embedding, Activation, BatchNormalization, GlobalAveragePooling1D, Input, merge, ZeroPadding1D
from keras.preprocessing import sequence
from keras.optimizers import RMSprop, Adam, SGD
from keras.regularizers import l2

#custom
print('import custom')
from import_data import scores ,test_scores
#uncomment if you need to vectorized new dataset
from vectorize import vectorized_train_titles, vectorized_test_titles

#recover titles already vectorize
#with open('data/vectorized_train.txt', 'r') as myfile:
    #data=vectorized_train_titles.read()#.replace('\n', '')
#with open('data/vectorized_test.txt', 'r') as myfile:
    #data=vectorized_test_titles.read()

#assigning train data
print('assigning train data')
print('y')
y_train = scores#np.array(scores)
print('x')
X_train = vectorized_train_titles #np.array(vectorized_train_titles)

#assigning train data
print('assigning test data')
X_test = vectorized_test_titles#np.array(vectorized_test_titles)
y_test = test_scores#np.array(test_scores)

#seting the model
print('setting the model')

model = Sequential()

model.add(Convolution1D(8, 2, W_regularizer=l2(0.005),input_shape=(110,300) ))
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
model.fit(X_train, y_train, validation_data=(X_test, y_test), batch_size=8, nb_epoch=50, shuffle=True, verbose=2)


#debug
print('model layer##################')
print(model.layers)
print('model sumary##################')
print(model.summary())




#model.save_weights("models/detector.finetuned.h5")

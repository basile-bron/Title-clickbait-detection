import tensorflow as tf
import pandas as pd
from keras.utils import np_utils  # from keras import utils as np_utils
import numpy as np
import matplotlib.pyplot as plt
import os
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

#for model usage in production
import tensorflowjs as tfjs

config = tf.compat.v1.ConfigProto(gpu_options =
                         tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction=0.8)
# device_count = {'GPU': 1}
)
config.gpu_options.allow_growth = True
session = tf.compat.v1.Session(config=config)
tf.compat.v1.keras.backend.set_session(session)


# parameters
vocab_size = 10000
embedding_dim = 16
max_length = 20
trunc_type = "post"
oov_tok = "<OOV>"
num_epochs = 5
verbose = 1
split=85

csv = pd.read_csv("titles.csv")[['titles', 'total']]
print(len(csv))
csv = csv.dropna()
print(len(csv))

assert not csv.isnull().values.any()

X = csv['titles']
Y = csv['total']
X.reset_index(drop=True, inplace=True)
Y.reset_index(drop=True, inplace=True)
X = X.astype(np.str)
print(X.shape)


def split_input(X, Y):
    """split the dataset to create the input data ( train_x, train_y, test_x, test_y )
    the size of the training/test set are define by a chosen pourcentage of the training dataset
    (e.g) 90% will generate a test set of 10% and training set of 90% of the original dataset """
    # (arbitrary)
    training_size_pourcentage = split
    end_x = int((len(X) / 100) * training_size_pourcentage)
    return X[:end_x], Y[:end_x], X[end_x:], Y[end_x:]


train_x, train_y, test_x, test_y = split_input(X, Y)

print(train_x[200])
print(train_y[200])

# you need to categorise before converting to one hot otherwise
# #it will get the last value of the dataset as the number of classes
"""Categorise data"""
train_y = pd.cut(train_y, bins=[-1, 26, np.inf], labels=[0, 1]).astype(np.float)
test_y = pd.cut(test_y, bins=[-1, 26, np.inf], labels=[0, 1]).astype(np.float)


tokenizer = Tokenizer(num_words=vocab_size,filters='"%&()*+,./:;<=>@[\\]^_`{|}~\t\n', lower=False, oov_token=oov_tok)
tokenizer.fit_on_texts(train_x)
word_index = tokenizer.word_index
sequences = tokenizer.texts_to_sequences(train_x)
padded_titles = pad_sequences(sequences, maxlen=max_length, padding='post')

test_sequences = tokenizer.texts_to_sequences(test_x)
test_padded_titles = pad_sequences(test_sequences, maxlen=max_length, padding='post')

reverse_word_index = dict([(value, key) for (key, value) in word_index.items()])


def decode_review(text):
    return ' '.join([reverse_word_index.get(i, '?') for i in text])


print(padded_titles.shape)
print(decode_review(padded_titles[200]))
print(train_y)

model = tf.keras.Sequential([
    tf.keras.layers.Embedding(vocab_size, embedding_dim, input_length=max_length),
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64, return_sequences=True)),
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(32,return_sequences=True)),
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(16)),
    tf.keras.layers.Dropout(0.5),

    #tf.keras.layers.Conv1D(128, 5, activation='relu'),
    #tf.keras.layers.GlobalMaxPooling1D(),
    #tf.keras.layers.Flatten(),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(1, activation='sigmoid')
])
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.summary()

#loading last  checkpoint
checkpoint_path = "checkpoint/"
try:
    pass
    #model = tf.keras.models.load_model(checkpoint_path)
except Exception as e:
    print(e)

# checkpoint
cp_callback = tf.keras.callbacks.ModelCheckpoint(
    checkpoint_path, verbose=1, save_best_only=True, period=2)  # save every 5 epoch


history = model.fit(padded_titles, train_y, epochs=num_epochs, validation_data=(test_padded_titles, test_y), callbacks=[cp_callback],
                    verbose=verbose)

#save model for production
tfjs.converters.save_keras_model(model, "modeljs/")

# list all data in history
print(history.history.keys())

# summarize history for accuracy and loss
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train_acc', 'test_acc'], loc='upper left')
plt.show()

# summarize history for accuracy and loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train_loss', 'test_loss'], loc='upper left')
plt.show()

predictions = model.predict(test_padded_titles)
titres = np.array(test_x)
for i in range(0, 200):
    print(titres[i])
    print(test_padded_titles[i])
    print(predictions[i])

#export word emb dimention for visualising on http://projector.tensorflow.org/
e = model.layers[0]
weights = e.get_weights()[0]
print(weights.shape) # shape: (vocab_size, embedding_dim)

import io

out_v = io.open('vecs.tsv', 'w', encoding='utf-8')
out_m = io.open('meta.tsv', 'w', encoding='utf-8')
for word_num in range(1, vocab_size):
  word = reverse_word_index[word_num]
  embeddings = weights[word_num]
  out_m.write(word + "\n")
  out_v.write('\t'.join([str(x) for x in embeddings]) + "\n")
out_v.close()
out_m.close()

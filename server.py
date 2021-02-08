from flask import Flask
from prediction import predict
import json

import tensorflow as tf
import pandas as pd
from keras.utils import np_utils  # from keras import utils as np_utils
import numpy as np
import matplotlib.pyplot as plt
import os
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

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

tokenizer = Tokenizer(num_words=vocab_size,filters='"%&()*+,./:;<=>@[\\]^_`{|}~\t\n', lower=False, oov_token=oov_tok)
tokenizer.fit_on_texts(train_x)

checkpoint_path = "checkpoint/"
try:
    pass
    model = tf.keras.models.load_model(checkpoint_path)
except Exception as e:
    print(e)


server = Flask(__name__)

@server.route("/")
def hello():
    return 'Title clickbait API'

@server.route("/api/<title>")
def api(title):
    result, test_sequences, test_padded_titles = predict(model, title, tokenizer, pad_sequences, max_length)
    list = [{'result': result, 'title': title},
            {'tokenized': test_sequences, 'padded': test_padded_titles}]
    for line in list:
        yield '%s</br>\n' % lineup
    return flask.Response(inner(), mimetype='text/html')

if __name__ == "__main__":
    server.run(host='0.0.0.0')

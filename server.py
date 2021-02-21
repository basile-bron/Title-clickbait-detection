from flask import Flask, Response
from prediction import predict
from flask_sslify import SSLify

import tensorflow as tf
from transformers import TFAutoModel, AutoTokenizer
import pandas as pd
import numpy as np

checkpoint_path = "checkpoint/"

model = tf.keras.models.load_model(checkpoint_path)

predictions, raw_outputs = model.predict(["Sam was a Wizard"])

server = Flask(__name__)
sslify = SSLify(server)


@server.route("/")
def hello():
    return 'Title clickbait API'


@server.route("/<title>")
def api(title):
    result, test_sequences, test_padded_titles = predict(model, title, tokenizer, pad_sequences, max_length)
    resp = Response(str(result))
    resp.headers['Access-Control-Allow-Origin'] = '*'
    return resp


if __name__ == "__main__":
    context = ('basile-bron.fr_2021-02-19.crt', 'basile-bron.fr_2021-02-19.key')  # certificate and key files
    server.run(host='0.0.0.0', ssl_context=context)

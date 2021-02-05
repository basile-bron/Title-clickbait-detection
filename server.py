from flask import Flask
from prediction import predict

server = Flask(__name__)

@server.route("/")
def hello():
    return 'lol'

@server.route("/api/<title>")
def api(title):
    result = predict(title)
    return ('%s') % result

if __name__ == "__main__":
    server.run(host='0.0.0.0')

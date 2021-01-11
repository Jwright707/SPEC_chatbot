import json
from spellchecker import SpellChecker
from nltk.stem.lancaster import LancasterStemmer
from flask import Flask
from flask_cors import CORS

from FlaskRoutes.route import chatbot_route
from Preprocessing.preprocessing_data import preprocessing_data
from NeuralNetwork.neural_network_layers import neural_network

spell = SpellChecker()
stemmer = LancasterStemmer()
with open("intents.json") as file:
    data = json.load(file)

# -------------- Preprocessing the data (Natural Language Processing) --------------------------------------------------

words, labels, training, output = preprocessing_data(data, stemmer)

# -------------- Creating the Neural network layers (Deep Learning) ----------------------------------------------------

model = neural_network(training, output)


app = Flask(__name__)
cors = CORS(app, resources={r"/*": {"origins": "*"}})


@app.route("/", methods=['POST'])
def chatbot():
    return chatbot_route(spell, model, words, stemmer, labels, data)


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=80, threaded=True, debug=False)

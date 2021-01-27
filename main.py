import json
import os

from spellchecker import SpellChecker
from nltk.stem.lancaster import LancasterStemmer
from flask import Flask
from flask_cors import CORS

from FlaskRoutes.route import chatbot_route, chatbot_answering
from Preprocessing.preprocessing_data import preprocessing_data
from NeuralNetwork.neural_network_layers import neural_network
from slack import WebClient
from dotenv import load_dotenv

load_dotenv()
spell = SpellChecker()
stemmer = LancasterStemmer()
with open("intents.json") as file:
    data = json.load(file)

slack_client = WebClient(token=os.getenv('SLACK_BOT_TOKEN'))

# -------------- Preprocessing the data (Natural Language Processing) --------------------------------------------------

words, labels, training, output = preprocessing_data(data, stemmer)

# -------------- Creating the Neural network layers (Deep Learning) ----------------------------------------------------

model = neural_network(training, output)

# -------------- Creating Flask Server ---------------------------------------------------------------------------------
app = Flask(__name__)
cors = CORS(app, resources={r"/*": {"origins": "*"}})

unidentified_questions = {}


@app.route("/", methods=['POST'])
def chatbot():
    return chatbot_route(spell, model, words, stemmer, labels, data, unidentified_questions, slack_client)


@app.route("/answer", methods=["POST"])
def answer():
    return chatbot_answering(unidentified_questions, slack_client)


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=80, threaded=True, debug=True)

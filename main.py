import json
import os
import threading
import time

from spellchecker import SpellChecker
from nltk.stem.lancaster import LancasterStemmer
from flask import Flask, Response
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

chatbot_helper = {
    "words": words,
    "labels": labels,
    "training": training,
    "output": output,
    "model": model,
    "data": data,
    'retraining': False
}


@app.route("/", methods=['POST'])
def chatbot():
    if chatbot_helper["retraining"]:
        return {
            "user_input": "Currently assisting another user, please ask me this question in 15 - 30 seconds. Sorry "
                          "for the inconvenience.",
            "context_state": "retraining"
        }
    else:
        return chatbot_route(spell, chatbot_helper['model'], chatbot_helper['words'], stemmer, chatbot_helper['labels'],
                             chatbot_helper['data'], unidentified_questions, slack_client)


@app.route("/answer", methods=["POST"])
def answer():
    chatbot_answering(unidentified_questions, slack_client, chatbot_helper, stemmer, spell)
    return Response(), 200


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=80, threaded=True, debug=True)

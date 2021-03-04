import json
import os

from spellchecker import SpellChecker
from nltk.stem.lancaster import LancasterStemmer
from flask import Flask, Response
from flask_cors import CORS

from FlaskRoutes.route import chatbot_route, chatbot_answering
from Helpers.slack_message_format import ignore_format
from Preprocessing.preprocessing_data import preprocessing_data
from NeuralNetwork.neural_network_layers import neural_network
from slack import WebClient
from dotenv import load_dotenv

load_dotenv()
stemmer = LancasterStemmer()
with open("intents.json") as file:
    data = json.load(file)

slack_client = WebClient(token=os.getenv('SLACK_BOT_TOKEN'))

# -------------- Preprocessing the data (Natural Language Processing) --------------------------------------------------

padded_training_x, padded_test_x, padded_training_y, padded_test_y, tokenizer, labels = preprocessing_data(data, stemmer)

# -------------- Creating the Neural network layers (Deep Learning) ----------------------------------------------------

model = neural_network(padded_training_x, padded_training_y, padded_test_x, padded_test_y, tokenizer)

# -------------- Creating Flask Server ---------------------------------------------------------------------------------
app = Flask(__name__)
cors = CORS(app, resources={r"/*": {"origins": "*"}})

unidentified_questions = {}

chatbot_helper = {
    "words": padded_training_x,
    "labels": labels,
    # "training": training,
    # "output": output,
    "model": model,
    "data": data,
    'retraining': False,
    'tokenizer': tokenizer
}


# Route communicates with the FE UI
@app.route("/", methods=['POST'])
def chatbot():
    if chatbot_helper["retraining"]:
        return {
            "user_input": "Currently assisting another user, please ask me this question in 15 - 30 seconds. Sorry "
                          "for the inconvenience.",
            "context_state": "retraining"
        }
    else:
        return chatbot_route(chatbot_helper['model'], chatbot_helper['words'], stemmer, chatbot_helper['labels'],
                             chatbot_helper['data'], unidentified_questions, slack_client, chatbot_helper['tokenizer'])


# Route communicates with Slack on responses to unidentified questions
@app.route("/answer", methods=["POST"])
def answer():
    chatbot_answering(unidentified_questions, slack_client, chatbot_helper, stemmer)
    return Response(), 200


# Route communicates with Slack on ignoring unidentified questions
@app.route("/ignore", methods=["POST"])
def ignore():
    questions = unidentified_questions
    first_key = list(questions.keys())[0]
    del unidentified_questions[first_key]
    ignore_response = ignore_format()
    # Sends the response to slack
    slack_client.chat_postMessage(**ignore_response)
    return Response(), 200


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=80, threaded=True, debug=True)

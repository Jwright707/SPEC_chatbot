import json
import os
import time

from slack.errors import SlackApiError
from spellchecker import SpellChecker
from nltk.stem.lancaster import LancasterStemmer
from flask import Flask, request, Response
from flask_cors import CORS

from FlaskRoutes.route import chatbot_route
from Helpers.slack_message_format import response_format, unidentified_format
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
    answer_request = request.form
    slack_response = answer_request["text"]
    questions = unidentified_questions
    if len(list(questions.keys())) >= 1:
        first_key = list(questions.keys())[0]
        unidentified_questions[first_key]['response'].append(slack_response)

        with open("test.json") as json_file:
            existing_data = json.load(json_file)

        existing_data['intents'].append(unidentified_questions[first_key])

        with open('test.json', 'w') as outfile:
            json.dump(existing_data, outfile, indent=4)

        del unidentified_questions[first_key]
        thank_you_response = response_format(slack_response)
        slack_client.chat_postMessage(**thank_you_response)
        if len(list(questions.keys())) >= 1:
            second_key = list(questions.keys())[0]
            user_question = unidentified_questions[second_key]['patterns'][0]
            slack_message = unidentified_format(user_question)
            slack_client.chat_postMessage(**slack_message)

    else:
        slack_client.chat_postMessage(channel=os.getenv("DOTTY_CHANNEL_ID"),
                                      text="There are currently no unidentified questions")
    return Response(), 200


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=80, threaded=True, debug=True)

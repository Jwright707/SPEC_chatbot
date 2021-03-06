import json
import os

from nltk.stem.lancaster import LancasterStemmer
from flask import Flask, Response
from flask_cors import CORS

from FlaskRoutes.route import chatbot_route, chatbot_answering
from Helpers.slack_message_format import ignore_format, no_question, unidentified_format
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
                             chatbot_helper['data'], unidentified_questions, slack_client)


# Route communicates with Slack on responses to unidentified questions
@app.route("/answer", methods=["POST"])
def answer():
    chatbot_answering(unidentified_questions, slack_client, chatbot_helper, stemmer)
    return Response(), 200


# Route communicates with Slack on ignoring unidentified questions
@app.route("/ignore", methods=["POST"])
def ignore():
    questions = unidentified_questions
    if len(questions) != 0:
        first_key = list(questions.keys())[0]
        del unidentified_questions[first_key]
        ignore_response = ignore_format()
        # Sends the response to slack
        slack_client.chat_postMessage(**ignore_response)
        if len(list(questions.keys())) >= 1:
            second_key = list(questions.keys())[0]
            user_question = unidentified_questions[second_key]['patterns'][0]
            slack_message = unidentified_format(user_question)
            slack_client.chat_postMessage(**slack_message)
    else:
        no_questions = no_question()
        slack_client.chat_postMessage(**no_questions)
    return Response(), 200


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=80, threaded=True, debug=True)

import json
import os
import threading
from flask import request, Response

from ChatBot.chat_bot import chat
from Helpers.slack_message_format import response_format, unidentified_format
from Helpers.string_cleaner import word_cleaner
from Preprocessing.retraining import retraining


def thread_function(index, chatbot_helper, stemmer):
    if index:
        return Response(), 200
    else:
        retraining(chatbot_helper, stemmer)


def chatbot_route(spell, model, words, stemmer, labels, data, unidentified_questions, slack_client):
    question_request = request.get_json()
    # User Input
    user_question = question_request["user_input"]
    # Context to the Input (For STM)
    context_state_user = question_request["context_state"]
    # Bag of Words Chatbot
    response = chat(
        spell, model, words, stemmer, labels, data,
        user_question, context_state_user, unidentified_questions, slack_client
    )
    return response


def chatbot_answering(unidentified_questions, slack_client, chatbot_helper, stemmer, spell):
    answer_request = request.form
    slack_response = answer_request["text"]
    questions = unidentified_questions
    if len(list(questions.keys())) >= 1:
        first_key = list(questions.keys())[0]
        # Adds user answer to the corresponding question
        unidentified_questions[first_key]['responses'].append(slack_response)

        # Loads in existing data
        with open("intents.json") as json_file:
            existing_data = json.load(json_file)

        # Appends the new object values to the existing array
        existing_data['intents'].append(unidentified_questions[first_key])

        with open('intents.json', 'w') as outfile:
            json.dump(existing_data, outfile, indent=2)

        # Deletes the old key from the object
        del unidentified_questions[first_key]
        # Formats the response to the slack user
        thank_you_response = response_format(slack_response)
        # Sends the response to slack
        slack_client.chat_postMessage(**thank_you_response)

        # Checks to see if there are more questions waiting to be answered
        if len(list(questions.keys())) >= 1:
            second_key = list(questions.keys())[0]
            user_question = unidentified_questions[second_key]['patterns'][0]
            slack_message = unidentified_format(user_question)
            slack_client.chat_postMessage(**slack_message)
        else:
            threads = list()
            for index in range(2):
                x = threading.Thread(target=thread_function, args=(index, chatbot_helper, stemmer))
                threads.append(x)
                x.start()
    else:
        slack_client.chat_postMessage(channel=os.getenv("DOTTY_CHANNEL_ID"),
                                      text="There are currently no unidentified questions")

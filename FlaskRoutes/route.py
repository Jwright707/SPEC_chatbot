from flask import request

from ChatBot.chat_bot import chat


def chatbot_route(spell, model, words, stemmer, labels, data):
    question_request = request.get_json()
    user_question = question_request["user_input"]
    context_state_user = question_request["context_state"]
    # Bag of Words Chatbot
    response = chat(spell, model, words, stemmer, labels, data, user_question, context_state_user)
    return response

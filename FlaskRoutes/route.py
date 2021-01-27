from flask import request

from ChatBot.chat_bot import chat


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

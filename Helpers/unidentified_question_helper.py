import json
import os

from Helpers.slack_message_format import unidentified_format


def unidentified_question_helper(unidentified_questions, joined_input, response, slack_client):
    unidentified_questions.append({
        'tag': "NEED TAG",
        'response': [joined_input],
        'patterns': [],
        'context_set': ''
    })
    response["response"] = "I don't understand, try asking a different question."
    response["context_state"] = ""
    with open('unidentified_questions.json', 'w') as outfile:
        json.dump(unidentified_questions, outfile, indent=4)
    slack_message = unidentified_format(joined_input)
    slack_client.chat_postMessage(**slack_message)
    return response

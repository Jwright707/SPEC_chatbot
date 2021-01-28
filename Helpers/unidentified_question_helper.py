import uuid

from Helpers.slack_message_format import unidentified_format
from Helpers.string_cleaner import word_cleaner


def unidentified_question_helper(unidentified_questions, joined_input, response, slack_client):
    question_id = str(uuid.uuid4())
    cleaned_input = word_cleaner(joined_input)
    unidentified_questions[question_id] = {
        'tag': question_id,
        'patterns': [cleaned_input],
        'responses': [],
        'context_set': ''
    }
    response["response"] = "I don't understand, try asking a different question."
    response["context_state"] = ""
    if len(list(unidentified_questions)) == 1:
        slack_message = unidentified_format(cleaned_input)
        slack_client.chat_postMessage(**slack_message)
    return response

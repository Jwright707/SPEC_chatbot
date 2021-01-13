def unidentified_question_helper(unidentified_questions, joined_input, response):
    unidentified_questions.append({
        'tag': "NEED TAG",
        'response': [joined_input],
        'patterns': [],
        'context_set': ''
    })
    response["response"] = "I don't understand, try asking a different question."
    response["context_state"] = ""
    return response

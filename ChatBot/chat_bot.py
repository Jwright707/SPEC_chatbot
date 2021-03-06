import random
import numpy as np
import nltk

from Helpers.string_cleaner import word_cleaner
from Helpers.unidentified_question_helper import unidentified_question_helper


def bag_of_words(user_input, words, stemmer):
    bag = [0 for _ in range(len(words))]

    s_words = nltk.word_tokenize(user_input)
    s_words = [stemmer.stem(word.lower()) for word in s_words]

    for se in s_words:
        for i, w in enumerate(words):
            if w == se:
                bag[i] = 1
    return np.array(bag)


def chat(model, words, stemmer, labels, data,
         user_question, context_state_user, unidentified_questions, slack_client):
    response = {}
    user_input = user_question

    cleaned_words = word_cleaner(user_input)
    results = model.predict([bag_of_words(cleaned_words, words, stemmer)])[0]
    # np.argmax gives the index of the greatest value in the list
    results_index = np.argmax(results)
    tag = labels[results_index]
    if cleaned_words == 'quit' or tag == 'goodbye':
        response["response"] = "Goodbye!"
        response["context_state"] = ""
        return response
    elif context_state_user == 'bug':
        response["response"] = "Thank you for reporting this issue/bug. We will work on fixing this."
        response["context_state"] = ""
        return response
    elif results[results_index] > 0.8 and context_state_user != 'bug':
        for tg in data["intents"]:
            if tg['tag'] == tag:
                if 'context_filter' not in tg or 'context_filter' in tg \
                        or 'context_set' not in tg or 'context_set' in tg \
                        and tg['context_filter'] == context_state_user:
                    responses = tg['responses']
                    if 'context_set' in tg:
                        context_state = tg['context_set']
                    else:
                        context_state = None
                    response["response"] = random.choice(responses)
                    response["context_state"] = context_state
                    return response
                else:
                    return unidentified_question_helper(unidentified_questions, cleaned_words, response, slack_client)
    else:
        return unidentified_question_helper(unidentified_questions, cleaned_words, response, slack_client)

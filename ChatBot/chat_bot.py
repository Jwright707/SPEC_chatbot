import json
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


def chat(spell, model, words, stemmer, labels, data, user_question, context_state_user, unidentified_questions):
    response = {}
    user_input = user_question
    corrected_user_input = [spell.correction(input_words).lower() for input_words in user_input.split()]
    joined_input = " ".join(corrected_user_input)
    results = model.predict([bag_of_words(word_cleaner(joined_input), words, stemmer)])[0]

    # np.argmax gives the index of the greatest value in the list
    results_index = np.argmax(results)
    tag = labels[results_index]

    if joined_input == 'quit' or tag == 'goodbye':
        with open('unidentified_questions.json', 'w') as outfile:
            json.dump(unidentified_questions, outfile, indent=4)
        response["response"] = "Goodbye!"
        response["context_state"] = ""
        return response
    elif context_state_user == 'bug':
        response["response"] = "Thank you for reporting this issue/bug. We will work on fixing this."
        response["context_state"] = ""
        return response
    elif results[results_index] > 0.7 and context_state_user != 'bug':
        for tg in data["intents"]:
            if tg['tag'] == tag:
                if 'context_filter' not in tg or 'context_filter' in tg and tg['context_filter'] == context_state_user:
                    responses = tg['responses']
                    if 'context_set' in tg:
                        context_state = tg['context_set']
                    else:
                        context_state = None
                    response["response"] = random.choice(responses)
                    response["context_state"] = context_state
                    return response
                else:
                    return unidentified_question_helper(unidentified_questions, joined_input, response)
    else:
        return unidentified_question_helper(unidentified_questions, joined_input, response)

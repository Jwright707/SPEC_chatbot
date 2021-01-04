import random
import numpy as np
import nltk

from Helpers.string_cleaner import word_cleaner


def bag_of_words(user_input, words, stemmer):
    bag = [0 for _ in range(len(words))]

    s_words = nltk.word_tokenize(user_input)
    s_words = [stemmer.stem(word.lower()) for word in s_words]

    for se in s_words:
        for i, w in enumerate(words):
            if w == se:
                bag[i] = 1
    return np.array(bag)


def chat(spell, model, words, stemmer, labels, data):
    print("Bot is ready to talk! (quit to stop)")
    # short term memory
    context_state = None
    while True:
        user_input = input("You: ")
        if user_input.lower() == "quit":
            break
        corrected_user_input = [spell.correction(input_words) for input_words in user_input.split()]
        joined_input = " ".join(corrected_user_input)
        results = model.predict([bag_of_words(word_cleaner(joined_input), words, stemmer)])[0]
        # np.argmax gives the index of the greatest value in the list
        results_index = np.argmax(results)
        tag = labels[results_index]

        if context_state == 'bug':
            print("Thank you for reporting this issue/bug. We will work on fixing this.")
            context_state = None
        elif results[results_index] > 0.7 and context_state != 'bug':
            for tg in data["intents"]:
                if tg['tag'] == tag:
                    # checks to see if the short term state is being used or not
                    if 'context_filter' not in tg or 'context_filter' in tg and tg['context_filter'] == context_state:
                        responses = tg['responses']
                        if 'context_set' in tg:
                            context_state = tg['context_set']
                        else:
                            context_state = None
                        print(random.choice(responses))
        else:
            print("I don't understand, try asking a different question.")

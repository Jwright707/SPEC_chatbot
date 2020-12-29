import os
import random
import re

import nltk
import numpy as np
import tflearn
import tensorflow as tf
import json
import pickle
from nltk.stem.lancaster import LancasterStemmer

stemmer = LancasterStemmer()
with open("intents.json") as file:
    data = json.load(file)


# --------------Preprocessing the data (Natural Language Processing) ---------------------------------------------------


def word_cleaner(string):
    string = string.lower()
    string = re.sub(r"i'm", "i am", string)
    string = re.sub(r"it's", "it is", string)
    string = re.sub(r"he's", "he is", string)
    string = re.sub(r"she's", "she is", string)
    string = re.sub(r"that's", "that is", string)
    string = re.sub(r"what's", "what is", string)
    string = re.sub(r"where's", "where is", string)
    string = re.sub(r"\'ll", " will", string)
    string = re.sub(r"\'ve", " have", string)
    string = re.sub(r"\'re", " are", string)
    string = re.sub(r"\'d", " would", string)
    string = re.sub(r"won't", "will not", string)
    string = re.sub(r"can't", "cannot", string)
    string = re.sub(r"[-()\"#/@;:<>{}+=~|.?,]", "", string)
    return string


try:
    with open("data.pickle", "rb") as f:
        words, labels, training, output = pickle.load(f)
except:
    words = []
    labels = []
    docs_x = []
    docs_y = []
    ignore_characters = ['!', '?', ',', '.']
    for intent in data["intents"]:
        for patten in intent["patterns"]:
            # stemmer takes each word in a pattern and reduces it to its root word
            # there? -> there or whats up -> what
            tokenized_words = nltk.word_tokenize(word_cleaner(patten))
            words.extend(tokenized_words)
            docs_x.append(tokenized_words)
            docs_y.append(intent["tag"])

        if intent["tag"] not in labels:
            labels.append(intent["tag"])

    # removes any duplicates
    words = [stemmer.stem(w.lower()) for w in words if w not in ignore_characters]
    words = sorted(list(set(words)))

    labels = sorted(labels)

    # Bag of words
    training = []
    output = []

    out_empty = [0 for _ in range(len(labels))]

    for x, doc in enumerate(docs_x):
        bag = []
        wrds = [stemmer.stem(w) for w in doc]

        for w in words:
            if w in wrds:
                bag.append(1)
            else:
                bag.append(0)

        output_row = out_empty[:]
        output_row[labels.index(docs_y[x])] = 1

        training.append(bag)
        output.append(output_row)

    # numpy.array takes the lists and turns them into arrays
    training = np.array(training)
    output = np.array(output)

    with open("data.pickle", "wb") as f:
        pickle.dump((words, labels, training, output), f)
# ----------------------------------------------------------------------------------------------------------------------

tf.reset_default_graph()

# --------------Creating the Neural network layers (Deep Learning) -----------------------------------------------------

# Input data which is the len of the training data
net = tflearn.input_data(shape=[None, len(training[0])])
# Have 8 neurons for a layer
net = tflearn.fully_connected(net, 8)
# Have 8 neurons for another layer
net = tflearn.fully_connected(net, 8)
# Have 8 neurons for the output layer (softmax gives a probability for each layer)
net = tflearn.fully_connected(net, len(output[0]), activation="softmax")
net = tflearn.regression(net)
# DNN is a type of neural network
model = tflearn.DNN(net)
model.fit(training, output, n_epoch=1000, batch_size=8, show_metric=True)
model.save("model.tflearn")
# ----------------------------------------------------------------------------------------------------------------------


# There is an error occuring here, this is temporary commented out
# try:
#     model.load("model.tflearn")
# except:
#     # Pass the training data (n_epoch is the number of times the model see the data)
#     model.fit(training, output, n_epoch=1000, batch_size=8, show_metric=True)
#     model.save("model.tflearn")


def bag_of_words(user_input, words):
    bag = [0 for _ in range(len(words))]

    s_words = nltk.word_tokenize(user_input)
    s_words = [stemmer.stem(word.lower()) for word in s_words]

    for se in s_words:
        for i, w in enumerate(words):
            if w == se:
                bag[i] = 1
    return np.array(bag)


def chat():
    print("Bot is ready to talk! (quit to stop)")
    # short term memory
    context_state = None
    while True:
        user_input = input("You: ")
        if user_input.lower() == "quit":
            break
        results = model.predict([bag_of_words(word_cleaner(user_input), words)])[0]
        # np.argmax gives the index of the greatest value in the list
        results_index = np.argmax(results)
        tag = labels[results_index]
        if results[results_index] > 0.7:
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


chat()

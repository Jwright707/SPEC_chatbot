import os

import nltk
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from nltk.corpus import stopwords
import pandas as pd
from future.moves import pickle
import numpy as np

from Helpers.string_cleaner import word_cleaner


def preprocessing_data(data, stemmer):
    # try:
    #     with open("data.pickle", "rb") as f:
    #         words, labels, training, output = pickle.load(f)
    #     return words, labels, training, output
    # except FileNotFoundError:
    words = []
    # Docs_x is an array of arrays of words
    docs_x = []
    # Docs_y is an array of tags for each array
    docs_y = []
    labels = []
    combine_patterns = []
    # combine_responses = []
    # combine_responses_set = set([])
    ignore_characters = ['!', '?', ',', '.']

    tokenizer = Tokenizer(oov_token="<OOV>")
    # nltk.download('stopwords')
    for word in set(stopwords.words('english')):
        ignore_characters.append(word)
    for intent in data["intents"]:
        for pat in range(len(intent["patterns"])):
            some_name = word_cleaner(intent["patterns"][pat])
            for replacer in ['!', '?', ',', '.']:
                some_name = some_name.replace(replacer, "")

            word_split = some_name.split(" ")
            words = [w.lower() for w in word_split if w not in ignore_characters]
            combine_patterns.append(" ".join(words))
            labels.append(intent['tag'])

    test_size = 0.7

    train_y = labels[:int(len(combine_patterns) * test_size)]
    train_x = combine_patterns[:int(len(combine_patterns) * test_size)]
    # train_y = labels
    # train_x = combine_patterns
    test_y = labels[int(len(combine_patterns) * test_size):]
    test_x = combine_patterns[int(len(combine_patterns) * test_size):]
    tokenizer.fit_on_texts(train_y)
    tokenizer.fit_on_texts(train_x)

    max_length = max([len(sentence.split(" ")) for sentence in combine_patterns])
    # sequences = tokenizer.texts_to_sequences(combine_patterns)

    # padded = pad_sequences(sequences, padding="pre", truncating='pre', maxlen=max_length)
    print(train_x)
    print(train_y)
    padded_training_x = pad_sequences(tokenizer.texts_to_sequences(train_x), padding="pre", truncating='pre',
                                      maxlen=max_length)
    padded_test_x = pad_sequences(tokenizer.texts_to_sequences(test_x), padding="pre", truncating='pre',
                                  maxlen=max_length)
    padded_training_y = pad_sequences(tokenizer.texts_to_sequences(train_y), padding="pre", truncating='pre',
                                      maxlen=max_length)
    padded_test_y = pad_sequences(tokenizer.texts_to_sequences(test_y), padding="pre", truncating='pre',
                                  maxlen=max_length)
    # print("Tokenizer:", tokenizer.word_index)
    # # print('Sequence', sequences)
    print('Padded Train X', padded_training_x)
    # print('Padded Test X', padded_test_x)
    print('Padded Train Y', padded_training_y)

    # print('Padded Test Y', padded_test_y)

    # if intent["tag"] not in labels:
    #     labels.append(intent["tag"])

    # removes any duplicates
    # stemmer takes each word in a pattern and reduces it to its root word
    # there? -> there or whats up -> what
    # words = [stemmer.stem(w.lower()) for w in words if w not in ignore_characters]
    # words = sorted(list(set(words)))
    # labels = sorted(labels)
    # # Bag of words
    # training = []
    # output = []
    #
    # # Array of 0 in the length of the labels
    # out_empty = [0 for _ in range(len(labels))]
    # for x, doc in enumerate(docs_x):
    #     # x is the key and doc is the array of words
    #     bag = []
    #     # Takes each word within each array of words and stems it
    #     words_to_bag = [stemmer.stem(w).lower() for w in doc]
    #     # Checks to see if there are any words within the words array that are in the stemmed words array
    #     for w in words:
    #         if w in words_to_bag:
    #             bag.append(1)
    #         else:
    #             bag.append(0)
    #     output_row = out_empty[:]
    #     output_row[labels.index(docs_y[x])] = 1
    #
    #     training.append(bag)
    #     output.append(output_row)
    #
    # # numpy.array takes the lists and turns them into arrays
    # training = np.array(training)
    # output = np.array(output)

    # with open("data.pickle", "wb") as f:
    #     pickle.dump((words, labels, training, output), f)
    return padded_training_x, padded_test_x, padded_training_y, padded_test_y, tokenizer, labels

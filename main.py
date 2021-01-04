import json
from spellchecker import SpellChecker
from nltk.stem.lancaster import LancasterStemmer

from ChatBot.chat_bot import chat
from Preprocessing.preprocessing_data import preprocessing_data
from NeuralNetwork.neural_network_layers import neural_network

spell = SpellChecker()
stemmer = LancasterStemmer()
with open("intents.json") as file:
    data = json.load(file)

# -------------- Preprocessing the data (Natural Language Processing) --------------------------------------------------

words, labels, training, output = preprocessing_data(data, stemmer)

# -------------- Creating the Neural network layers (Deep Learning) ----------------------------------------------------

model = neural_network(training, output)

# -------------- Bag of Words Chatbot ----------------------------------------------------------------------------------
chat(spell, model, words, stemmer, labels, data)

import nltk
from nltk.corpus import stopwords
from future.moves import pickle
import numpy as np

from Helpers.string_cleaner import word_cleaner


def preprocessing_data(data, stemmer):
    try:
        with open("data.pickle", "rb") as f:
            words, labels, training, output = pickle.load(f)
        return words, labels, training, output
    except FileNotFoundError:
        words = []
        labels = []
        docs_x = []
        docs_y = []
        ignore_characters = ['!', '?', ',', '.']
        # nltk.download('stopwords')
        for word in set(stopwords.words('english')):
            ignore_characters.append(word)
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
            words_to_bag = [stemmer.stem(w).lower() for w in doc]

            for w in words:
                if w in words_to_bag:
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

        return words, labels, training, output

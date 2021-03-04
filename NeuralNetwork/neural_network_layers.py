import os

import tflearn
import tensorflow as tf
import numpy as np


def neural_network(padded_training_x, padded_training_y, padded_test_x, padded_test_y, tokenizer):
    tf.reset_default_graph()
    # Input data which is the len of the training data
    net = tflearn.input_data(shape=[None, len(padded_training_x[0])])
    # Have 8 neurons for a layer
    net = tflearn.fully_connected(net, 8)
    # Have 8 neurons for another layer
    net = tflearn.fully_connected(net, 8)
    # Have 8 neurons for the output layer (softmax gives a probability for each layer)
    net = tflearn.fully_connected(net, len(padded_training_y[0]), activation="softmax")
    net = tflearn.regression(net)
    # DNN is a type of neural network
    model = tflearn.DNN(net)

    model.fit(padded_training_x, padded_training_y, validation_set=(padded_test_x, padded_training_y), n_epoch=50, batch_size=8, show_metric=True)
    # print(training[0])
    # print(output[0])
    # print(padded_test_x[0])
    # print(np.array(padded_test_x[0]))
    # input_value = padded_test_x[0][None, :]
    # print(model.predict(input_value))
    model.save("model.tflearn")

    return model

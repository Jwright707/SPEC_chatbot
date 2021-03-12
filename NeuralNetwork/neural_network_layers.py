import os

import tflearn
import tensorflow as tf
import numpy as np
import sklearn
from sklearn.svm import SVC


class EarlyStoppingCallback(tflearn.callbacks.Callback):
    def __init__(self, val_acc_thresh):
        """ Note: We are free to define our init function however we please. """
        self.val_acc_thresh = val_acc_thresh

    def on_epoch_end(self, training_state):
        """ """
        # Apparently this can happen.

        if training_state.val_acc is None: return
        if training_state.val_acc > self.val_acc_thresh:
            raise StopIteration


def neural_network(padded_training_x, padded_training_y, padded_test_x, padded_test_y, tokenizer):
    model = SVC()
    model.fit(padded_training_x, padded_training_y)
    # tf.reset_default_graph()
    # early_stopping_cb = EarlyStoppingCallback(val_acc_thresh=0.5)
    #
    # # Input data which is the len of the training data
    # net = tflearn.input_data(shape=[None, len(padded_training_x[0])])
    # # Have 8 neurons for a layer
    # net = tflearn.fully_connected(net, 8)
    # # Have 8 neurons for another layer
    # net = tflearn.fully_connected(net, 8)
    # # Have 8 neurons for the output layer (softmax gives a probability for each layer)
    # net = tflearn.fully_connected(net, 8, activation="softmax")
    # net = tflearn.regression(net, optimizer='adam', learning_rate=0.01, loss='mean_square')
    # # DNN is a type of neural network
    # model = tflearn.DNN(net)
    #
    # model.fit(padded_training_x, padded_training_y,
    #           # validation_set=(padded_test_x, padded_test_y),
    #           n_epoch=20000, batch_size=8, show_metric=True,
    #           callbacks=early_stopping_cb,
    #           shuffle=True)
    # # print(training[0])
    # # print(output[0])
    # # print(padded_test_x[0])
    # # print(np.array(padded_test_x[0]))
    # # input_value = padded_test_x[0][None, :]
    # # print(model.predict(input_value))
    # model.save("model.tflearn")

    return model

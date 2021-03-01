import os

import tflearn
import tensorflow as tf


def neural_network(training, output, vocab_size):
    # model = tf.keras.Sequential([
    #     tf.keras.layers.Embedding("", "", ""),
    #     tf.keras.layers.GlobalAveragePooling1D(),
    #     tf.keras.layers.Dense(24, activation='relu'),
    #     tf.keras.layers.Dense(1, activation='sigmoid')
    # ])
    # model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    tf.reset_default_graph()
    # Input data which is the len of the training data
    net = tflearn.input_data(shape=[None, len(training[0])])
    # net = tflearn.embedding(net, input_dim=vocab_size, output_dim=len(training[0]))
    # Have 8 neurons for a layer
    net = tflearn.fully_connected(net, 8)
    # Have 8 neurons for another layer
    net = tflearn.fully_connected(net, 8)
    # Have 8 neurons for the output layer (softmax gives a probability for each layer)
    net = tflearn.fully_connected(net, len(output[0]), activation="softmax")
    net = tflearn.regression(net)
    # DNN is a type of neural network
    model = tflearn.DNN(net)

    # model.fit(training, output, n_epoch=1000, batch_size=8, show_metric=True)
    # model.save("model.tflearn")

    # There is an error occurring here, this is temporary commented out
    if os.path.exists("model.tflearn.meta"):
        model.load("model.tflearn")
    else:
        model.fit(training, output, n_epoch=1000, batch_size=8, show_metric=True)
        model.save("model.tflearn")

    return model

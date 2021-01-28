import tflearn
import tensorflow as tf


def retraining_neural_network(training, output):
    tf.reset_default_graph()
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

    model.fit(training, output, n_epoch=300, batch_size=8, show_metric=True)
    model.save("model.tflearn")
    return model

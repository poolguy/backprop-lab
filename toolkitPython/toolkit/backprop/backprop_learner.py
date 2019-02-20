import math
import numpy as np

from toolkitPython.toolkit.backprop.network import Layer
from toolkitPython.toolkit.backprop.network import OutputLayer

from toolkitPython.toolkit.supervised_learner import SupervisedLearner
# Create structure

# Initialise all weights to small random values

# For each input vector
#   Forward phase:
#

# Implement momentum LR

class BackpropLearner(SupervisedLearner):
    def __init__(self, n_hidden_layers, n_hidden_nodes, n_output_nodes):
        self.n_hidden_layers = n_hidden_layers
        self.n_hidden_nodes = n_hidden_nodes
        self.n_output_nodes = n_output_nodes
        self.layers = []

    def init_network(self, n_inputs, output_class_dict):
        # first hidden layer (needs to be initialized with number of weights equal to inputs
        self.layers.append(Layer(self.n_hidden_nodes, n_inputs+1))
        # additional hidden layers
        for i in range(1, self.n_hidden_layers):
            self.layers.append(Layer(self.n_hidden_nodes, self.n_hidden_nodes+1))
        # output layer
        self.layers.append(OutputLayer(self.n_output_nodes, self.n_hidden_nodes+1, output_class_dict))

    def add_bias_if_necessary(self, row):
        if len(row) != self.len_with_bias:
            row.append(1)

    def train(self, features, labels):
        # initialize network
        self.init_network(features.cols, labels.enum_to_str[0])
        self.len_with_bias = len(features.data[0]) + 1

        # for each row of data, train the network all the way through, and propagate the error
        for row in features.data:
            # initialize input row
            self.add_bias_if_necessary(row)
            inputs = np.array(row)

            # input moves forward through the network
            for layer in self.layers:
                layer.set_inputs(inputs)
                outputs = layer.get_outputs()
                inputs = outputs



            # error moves backward through the network

            pass

    def predict(self, features, labels):
        pass

    def measure_accuracy(self, features, labels, confusion=None):
        return super().measure_accuracy(features, labels, confusion)




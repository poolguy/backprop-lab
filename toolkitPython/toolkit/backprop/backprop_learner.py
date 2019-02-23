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
    def __init__(self, n_hidden_layers, n_hidden_nodes, n_output_nodes, lr, alpha):
        self.n_hidden_layers = n_hidden_layers
        self.n_hidden_nodes = n_hidden_nodes
        self.n_output_nodes = n_output_nodes
        self.layers = []
        self.lr = lr
        self.alpha = alpha

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

    # todo: stopping criteria

    def train(self, features, labels):
        # initialize network
        self.init_network(features.cols, labels.enum_to_str[0])
        self.len_with_bias = len(features.data[0]) + 1

        # for each row of data, train the network all the way through, and propagate the error
        for i, row in enumerate(features.data):
            # initialize input row
            self.add_bias_if_necessary(row)
            inputs = np.array(row)

            # input moves forward through the network
            for layer in self.layers:
                # todo: may wish to rethink how to handle inputs and outputs
                layer.set_inputs(inputs)
                outputs = layer.get_outputs()
                inputs = outputs

            # error moves backward through the network, updating weights as it goes
            target_class = labels.enum_to_str[0][labels.data[i][0]]
            for layer in reversed(self.layers):
                # update weights
                if type(layer) is OutputLayer:
                    deltas, weights = layer.update_weights_and_get_deltas_and_weights_o(target_class, self.lr, self.alpha)
                else:
                    deltas, weights = layer.update_weights_and_get_deltas_and_weights(deltas, weights, self.lr, self.alpha)

            # kinda lazy, but simple solution to removing lingering member variables
            for layer in self.layers:
                layer.scrub_lingering_member_variables()


    def predict(self, features, labels):
        pass

    def measure_accuracy(self, features, labels, confusion=None):
        return super().measure_accuracy(features, labels, confusion)




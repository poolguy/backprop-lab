import math

import numpy as np


class Layer:
    def __init__(self, n_nodes, n_weights):
        self.nodes = []
        for i in range(n_nodes):
            self.nodes.append(Node(n_weights))
        self.nodes.append(BiasNode(n_weights))

    def set_inputs(self, inputs):
        self.inputs = inputs

# class representing a node
# Note: the node class contains all incoming weights to the node, not outgoing weights
class Node:
    def __init__(self, n_weights):
        self.weights = np.random.uniform(-1,1,size=n_weights)
        self.output = None
        self.net = None
        self.delta = None
        self.inputs = None

    def set_inputs(self, inputs):
        self.inputs = inputs

    def get_inputs(self):
        return self.inputs

    # gets or computes the output
    def get_output(self):
        if self.output is not None:
            return self.output
        else:
            return self.sigmoid(self.get_net())

    # gets or computes the net value for the node
    def get_net(self):
        if self.net is not None:
            return self.net
        else:
            return self.compute_net()

    # computes net
    def compute_net(self):
        return np.dot(self.weights, self.inputs)

    # gets or computes delta for the node
    def get_delta(self, deltas_and_weights):
        if self.delta is not None:
            return self.delta
        else:
            return self.compute_delta(deltas_and_weights)

    # args:
    ## deltas_and_weights:
    ### a list of tuples containing all of the deltas from the following layer in the network with their corresponding weights
    # this must be called after net is computed
    def compute_delta(self, deltas_and_weights):
        delta = 0
        for d, w in deltas_and_weights:
            delta += d * w * self.dx_sigmoid(self.get_net())
        return delta

    # compute weight updates
    # Must be called after delta is computed for the node
    def compute_weight_updates(self, lr):
        updates = np.zeros(shape=self.weights.shape)
        for i, Z in enumerate(self.inputs):
            updates[i] = lr * self.get_delta(None) * Z

    def sigmoid(self, net):
        return 1 / (1 + (math.e ** (-net)))

    def dx_sigmoid(self, net):
        return self.sigmoid(net)(1 - self.sigmoid(net))

class BiasNode(Node):
    def get_output(self):
        return 1


class OutputNode(Node):

    def get_output(self):
        return np.dot(self.weights, self.inputs)

    # todo: net on output nodes?
    def get_delta(self, target):
        return


class OutputLayer:
    pass
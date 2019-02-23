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

    # returns np array of outputs from the layer
    def get_outputs(self):
        outputs = []
        for node in self.nodes:
            node.set_inputs(self.inputs)
            outputs.append(node.get_output())
        return np.array(outputs)

    def update_weights_and_get_deltas_and_weights(self, k_deltas, k_weights, lr, alpha):
        deltas = []
        weights = []
        for i, node in enumerate(self.nodes):
            jk_weights = np.array(k_weights)[:,i]
            deltas.append(node.compute_and_get_delta(k_deltas, jk_weights))
            weights.append(node.weights)
            # after delta has been computed for the node, update the weights
            node.update_weights(lr, alpha)

        return deltas, weights

    # This is called after a full forward pass of the network, and subsequent backpropagation of the error
    def scrub_lingering_member_variables(self):
        for node in self.nodes:
            node.scrub_lingering_member_variables()

class OutputLayer(Layer):
    def __init__(self, n_nodes, n_weights, output_class_dict):
        super().__init__(0, 0)
        self.nodes = []
        for i in range(n_nodes):
            self.nodes.append(OutputNode(n_weights, output_class_dict[i]))

    def update_weights_and_get_deltas_and_weights_o(self, target_class, lr, alpha):
        deltas = []
        weights = []
        for node in self.nodes:
            deltas.append(node.compute_and_get_delta_o(target_class))
            weights.append(node.weights)
            # after delta has been computed for the node, update the weights
            node.update_weights(lr, alpha)

        return deltas, weights

# class representing a node
# Note: the node class contains all incoming weights to the node, not outgoing weights
class Node:
    def __init__(self, n_weights):
        self.weights = np.random.uniform(-1,1,size=n_weights)
        self.previous_updates = np.zeros(shape=self.weights.shape) # this is for momentum
        self.output = None
        self.net = None
        self.delta = None
        self.inputs = None

    # Forward: 1
    def set_inputs(self, inputs):
        self.inputs = inputs

    def get_inputs(self):
        return self.inputs

    # Forward: 2a
    # gets or computes the output todo:
    def get_output(self):
        if self.output is None:
            self.output = self.sigmoid(self.get_net())
        return self.output

    # Forward: 2b
    # gets or computes the net value for the node todo:
    def get_net(self):
        if self.net is None:
            self.net = self.compute_net()
        return self.net

    # Forward: 2c
    # computes net
    def compute_net(self):
        return np.dot(self.weights, self.inputs)

    # Backward:
    # gets or computes delta for the node. Assumes that compute_and_get_delta has already been called.
    def get_delta(self):
        return self.delta

    # args:
    ## deltas_and_weights:
    ### deltas: a list of containing all of the deltas from the following layer in the network
    ### weights: a list of weights connecting the current node to each of the nodes corresponding with the deltas from the following layer
    # This must be called after net is computed.
    # These two lists should always be the same size. If they aren't, something is wrong. Each delta corresponds to a weight in the weight list with the same index.
    def compute_and_get_delta(self, deltas, weights):
        self.delta = 0
        for i in range(len(deltas)):
            d = deltas[i]
            w = weights[i]
            self.delta += d * w * self.dx_sigmoid(self.get_net())
        return self.delta

    # updates the weights of the node. Called after delta has been computed.
    def update_weights(self, lr, alpha):
        weight_updates = self.compute_weight_updates(lr, alpha)
        self.weights = self.weights + weight_updates

    # compute weight updates
    # Must be called after delta is computed for the node
    def compute_weight_updates(self, lr, alpha):
        updates = np.zeros(shape=self.weights.shape)
        for i, Z in enumerate(self.inputs):
            # implements momentum in weight update
            updates[i] = (lr * self.get_delta() * Z) + (alpha * self.previous_updates[i])

        self.previous_updates = updates
        return updates

    def sigmoid(self, net):
        return 1 / (1 + (math.e ** (-net)))

    def dx_sigmoid(self, net):
        return self.sigmoid(net)*(1 - self.sigmoid(net))

    # This is called after a full forward pass of the network, and subsequent backpropagation of the error
    def scrub_lingering_member_variables(self):
        self.output = None
        self.net = None
        self.delta = None


class BiasNode(Node):
    def get_output(self):
        return 1


class OutputNode(Node):
    def __init__(self, n_weights, output_class):
        super().__init__(n_weights)
        self.output_class = output_class

    def compute_and_get_delta_o(self, target_class):
        target = self.enumerate_target_class(target_class)
        self.delta = (target - self.get_output()) * self.dx_sigmoid(self.get_net())
        return self.delta

    def enumerate_target_class(self, target_class):
        if target_class == self.output_class:
            return 1
        else:
            return 0
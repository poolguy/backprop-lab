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
        self.final_labels = []
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

    # todo: stopping criteria, predict, measure_accuracy?
    def train(self, features, labels):
        # initialize network
        self.target_class_dict = labels.str_to_enum[0]
        self.init_network(features.cols, labels.enum_to_str[0])
        # todo remove test: init weights for test
        # self.init_weights_for_test()
        self.len_with_bias = len(features.data[0]) + 1
        self.training = True

        ## Stopping criteria management
        best_accuracy = 0
        n_epochs = 0
        n_epochs_without_improvement = 0
        while n_epochs_without_improvement < 5:
            n_epochs += 1
            features.shuffle(labels)

            # todo remove test: Before each pattern presentation, print the weights
            # self.print_weights()

            # for each row of data, train the network all the way through, and propagate the error
            for i, row in enumerate(features.data):
                # initialize input row
                self.add_bias_if_necessary(row)
                inputs = np.array(row)

                # input moves forward through the network
                for layer in self.layers:
                    layer.set_inputs(inputs)
                    outputs = layer.get_outputs()
                    inputs = outputs

                # todo remove test: After you forward-propagate the input vector, print the predicted output vector
                # print("Predicted Output: ", outputs)

                # outputs = self.round_outputs(outputs)

                try:
                    # target = np.zeros(shape=self.n_output_nodes) # categorical targets
                    # target[int(labels.data[i][0])] = 1
                    target = labels.enum_to_str[0][int(labels.data[i][0])]
                except:
                    target = labels.data[i] # continuous targets
                # error moves backward through the network, updating weights as it goes
                for layer in reversed(self.layers):
                    # update weights
                    if type(layer) is OutputLayer:
                        deltas, weights = layer.update_weights_and_get_deltas_and_weights_o(target, self.lr, self.alpha)
                    else:
                        deltas, weights = layer.update_weights_and_get_deltas_and_weights(deltas, weights, self.lr, self.alpha)

                # todo remove test: print the error values assigned to each network unit
                # self.print_errors()
                # if n_epochs_without_improvement > 98:
                #     self.print_weights()
                #     print("Predicted Output: ", outputs)
                #     self.print_errors()

                # kinda lazy, but simple solution to removing lingering member variables within the network
                self.scrub_network()


            ## Stopping criteria management
            # Track if accuracy has improved
            accuracy = self.measure_accuracy(features, labels)
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                self.best_network = self.layers.copy()
                n_epochs_without_improvement = 0
            else:
                n_epochs_without_improvement += 1

        self.training = False
        self.measure_accuracy(features,labels)

        print(self.final_labels)


    def predict(self, features, labels):
        del labels[:]

        # initialize input row
        self.add_bias_if_necessary(features)
        inputs = np.array(features)

        if self.training:
            network = self.layers
        else:
            network = self.best_network

        # input moves forward through the network
        for layer in network:
            layer.set_inputs(inputs)
            outputs = layer.get_outputs()
            inputs = outputs

        prediction = self.target_class_dict[layer.get_prediction()]
        # prediction = self.round_outputs(outputs)
        labels.append(prediction)

        # if not self.training:
        self.final_labels.append(prediction)
        self.scrub_network()

    def init_weights_for_test(self):
        self.layers[0].nodes[0].weights = np.array([.2,-.1,.1])
        self.layers[0].nodes[1].weights = np.array([.3,-.3,-.2])
        self.layers[1].nodes[0].weights = np.array([-.2,-.3,.1])
        self.layers[1].nodes[1].weights = np.array([-.1,.3,.2])
        self.layers[2].nodes[0].weights = np.array([-.1,.3,.2])
        self.layers[2].nodes[1].weights = np.array([-.2,-.3,.1])

    def print_weights(self):
        print("Weights:")
        for layer in self.layers:
            for node in layer.nodes:
                print("\t", node.weights)

    def print_errors(self):
        print("Error Values:")
        for layer in self.layers:
            for node in layer.nodes:
                print("\t", node.delta)

    def round_outputs(self, outputs):
        largest = 0
        largest_idx = None
        for i, output in enumerate(outputs):
            if output > largest:
                largest = output
                largest_idx = i

        outputs = np.zeros(shape=outputs.shape)
        outputs[largest_idx] = 1
        return outputs
    
    # kinda lazy, but simple solution to removing lingering member variables within the network
    def scrub_network(self):
        for layer in self.layers:
            layer.scrub_lingering_member_variables()
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
        # members for reporting todo: from step 2
        # self.train_mses = [] todo: from step 2
        # self.vs_mses = [] todo: from step 2
        # self.accuracies = [] todo: from step 2

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
        self.target_class_dict = labels.str_to_enum[0]
        self.init_network(features.cols, labels.enum_to_str[0])
        self.len_with_bias = len(features.data[0]) + 1
        self.training = True

        ## Stopping criteria management
        best_vs_mse = math.inf
        n_epochs = 0
        n_epochs_without_improvement = 0
        # Pull out validation set
        vs_features, vs_labels = self.get_train_and_validation_set(features, labels)
        while n_epochs_without_improvement < 10:
            features.shuffle(labels)
            n_epochs += 1
            # train_sse = 0 todo: from step 2
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

                try:
                    target = labels.enum_to_str[0][int(labels.data[i][0])]
                except:
                    target = labels.data[i] # continuous targets
                # error propagates backward through the network, updating weights as it goes
                for layer in reversed(self.layers):
                    # update weights
                    if type(layer) is OutputLayer:
                        deltas, weights = layer.update_weights_and_get_deltas_and_weights_o(target, self.lr, self.alpha)
                        # train_sse += layer.compute_sse(target) todo: from step 2
                    else:
                        deltas, weights = layer.update_weights_and_get_deltas_and_weights(deltas, weights, self.lr, self.alpha)

                # kinda lazy, but simple solution to removing lingering member variables within the network
                self.scrub_network()

            # train_mse = train_sse/features.rows todo: from step 2
            # self.train_mses.append(train_mse) todo: from step 2
            # accuracy = self.measure_accuracy(vs_features, vs_labels) todo: from step 2
            # self.accuracies.append(accuracy) todo: from step 2

            ## Stopping criteria management
            # Track if vs_mse has improved
            vs_mse = self.measure_mse(vs_features, vs_labels)
            # self.vs_mses.append(vs_mse) todo: from step 2
            if vs_mse < best_vs_mse:
                best_vs_mse = vs_mse
                self.best_network = self.layers.copy()
                n_epochs_without_improvement = 0
            else:
                n_epochs_without_improvement += 1

        self.training = False
        train_acc = self.measure_accuracy(features,labels)

        # self.print_list("Train MSEs", self.train_mses) todo: from step 2
        # self.print_list("VS MSEs", self.vs_mses) todo: from step 2
        # self.print_list("Accuracies", self.accuracies) todo: from step 2

        # Step 3: Compute MSE for train, test, and validation
        train_mse = self.measure_mse(features, labels)
        test_mse = self.measure_mse(self.test_features, self.test_labels)
        test_acc = self.measure_accuracy(self.test_features, self.test_labels)

        print(self.n_hidden_layers)
        print(train_mse)
        print(test_mse)
        print(vs_mse)
        print(n_epochs)
        print(train_acc)
        print(test_acc)


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

        if not self.training:
            self.final_labels.append(prediction)
        self.scrub_network()

    def measure_mse(self, features, labels):
        sse = 0
        for i, row in enumerate(features.data):
            # initialize input row
            self.add_bias_if_necessary(row)
            inputs = np.array(row)

            # input moves forward through the network
            for layer in self.layers:
                layer.set_inputs(inputs)
                outputs = layer.get_outputs()
                inputs = outputs

            try:
                target = labels.enum_to_str[0][int(labels.data[i][0])]
            except:
                target = labels.data[i]  # continuous targets

            sse += layer.compute_sse(target)

            self.scrub_network()

        mse = sse/features.rows
        return mse

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

    def get_train_and_validation_set(self, features, labels):
        vs_labels = labels.__copy__()
        vs_features = features.__copy__()

        d = features.data
        vs_features.data = list(d[::10])
        features.data = list([list(d[i]) for i in range(len(d)) if i%10 != 0])

        l = labels.data
        vs_labels.data = list(l[::10])
        labels.data = list([list(l[i]) for i in range(len(l)) if i%10 != 0])
        return vs_features, vs_labels

    def print_list(self, name, list):
        print("\n", name)
        for item in list:
            print(item)
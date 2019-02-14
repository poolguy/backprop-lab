import math

from toolkitPython.toolkit.supervised_learner import SupervisedLearner
# Create structure

# Initialise all weights to small random values

# For each input vector
#   Forward phase:
#

# Implement momentum LR

class BackpropLearner(SupervisedLearner):
    def __init__(self, num_hidden_layers, num_hidden_nodes):
        self.num_hidden_layers = num_hidden_layers
        self.num_hidden_nodes = num_hidden_nodes

    def train(self, features, labels):
        # initialize network


        pass

    def predict(self, features, labels):
        pass

    def measure_accuracy(self, features, labels, confusion=None):
        return super().measure_accuracy(features, labels, confusion)




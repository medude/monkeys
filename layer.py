from functions import *


class Layer:
    def __init__(self, value, neuron_num=3, is_output=False, bias=0):
        self.value = value
        self.neuron_num = neuron_num
        self.is_output = is_output
        self.bias = bias
        self.error = 0
        self.delta = 0

    def calc_error(self, next_layer, synapse=None):
        if self.is_output:
            self.error = next_layer - self.value
            return self.error
        else:
            self.error = next_layer.delta.dot(synapse.value.T)
            return self.error

    def calc_delta(self):
        self.delta = self.error * nonlin(self.value, deriv=True)
        return self.delta

from functions import *


class SynapseSet:
    def __init__(self, prev_layer=None, next_layer=None):
        self.prev_layer = prev_layer
        self.next_layer = next_layer

        self.value = 2 * np.random.random((
            prev_layer["neuron_num"] + prev_layer["bias"],
            next_layer["neuron_num"] + next_layer["bias"]
        )) - 1

    def set_layers(self, prev_layer, next_layer, calc_value=False):
        self.prev_layer = prev_layer
        self.next_layer = next_layer

        if calc_value:
            self.value = 2 * np.random.random((
                prev_layer.neuron_num + prev_layer.bias,
                next_layer.neuron_num + next_layer.bias
            )) - 1

    def update_weights(self):
        self.value += self.prev_layer.value.T.dot(self.next_layer.delta)

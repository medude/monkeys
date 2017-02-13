# This code comes from a demo NN program from the YouTube video https://youtu.be/h3l4qz76JhQ. The program creates a
# neural network that simulates the exclusive OR function with two inputs and one output.

import json
from matplotlib import pyplot

from layer import *
from synapse_set import *
from functions import *

# The following is a function definition of the sigmoid function, which is the type of non-linearity chosen for this
# neural net. It is not the only type of non-linearity that can be chosen, but is has nice analytical features and is
# easy to teach with. In practice, large-scale deep learning systems use piecewise-linear functions because they are
# much less expensive to evaluate.
#
# The implementation of this function does double duty. If the deriv=True flag is passed in, the function instead
# calculates the derivative of the function, which is used in the error backpropogation step.

# input and output data
with open("res/us_pop_train.json", "r") as data_file:
    data = json.loads(data_file.read())
X = np.array(data[0])
y = np.array(data[1])

# make it deterministic
np.random.seed(1)

# neuron layers
neuron_layers = [
    Layer(None, None, neuron_num=1, bias=1),
    Layer(None, None, neuron_num=10),
    Layer(None, None, neuron_num=2, is_output=True)
]

# synapses
synapse_sets = [
    SynapseSet(prev_layer=neuron_layers[0], next_layer=neuron_layers[1]),
    SynapseSet(prev_layer=neuron_layers[1], next_layer=neuron_layers[2])
]

# training step
for j in range(240000):
    # for j in range(10):
    # Calculate forward through the network.
    neuron_layers[0] = Layer(X, synapse_sets[0], neuron_num=1, bias=1)
    neuron_layers[1] = Layer(nonlin(np.dot(neuron_layers[0].value, synapse_sets[0].value)), synapse_sets[1],
                             neuron_num=10)
    neuron_layers[2] = Layer(nonlin(np.dot(neuron_layers[1].value, synapse_sets[1].value)), None, is_output=True)

    # Backpropagation of errors using the chain rule.
    neuron_layers[2].calc_error(y)
    neuron_layers[2].calc_delta()
    neuron_layers[1].calc_error(neuron_layers[2])
    neuron_layers[1].calc_delta()

    # Sometimes print out the error
    if (j % 10000) == 0:
        print("Error: " + str(np.mean(np.abs(neuron_layers[2].error))))

    synapse_sets[0].set_layers(neuron_layers[0], neuron_layers[1])
    synapse_sets[1].set_layers(neuron_layers[1], neuron_layers[2])

    # update weights (no learning rate term)
    synapse_sets[1].update_weights()
    synapse_sets[0].update_weights()

print("Output after training")
print(neuron_layers[2].value)

print("Weights")

print("Synapse 0: " + str(synapse_sets[0].value) + "\n")
print("Synapse 1: " + str(synapse_sets[1].value))

X = X.flatten("F")[:X.shape[0]]

graph_x = np.array(list(frange(X.min(), X.max(), (X.max() - X.min()) / X.size)))
graph_y = np.array([])

for i in range(graph_x.size):
    l0 = np.array([
        [graph_x[i], 1]
    ])
    l1 = np.array(nonlin(np.dot(l0, synapse_sets[0].value)))
    l2 = np.array(nonlin(np.dot(l1, synapse_sets[1].value)))

    graph_y = np.append(graph_y, l2.mean())

pyplot.title("U.S. Population")
pyplot.xlabel("Year")
pyplot.ylabel("Population")

pyplot.plot(graph_x * 200 + 1900, graph_y * 400000000 + 76094000, c="g", label="Prediction")
pyplot.scatter(X * 200 + 1900, y * 400000000 + 76094000, c="b", label="Actual")

pyplot.legend()
pyplot.show()

while True:
    test1 = float(input("First number:  "))

    l0 = np.array([
        [test1, 1]
    ])
    l1 = np.array(nonlin(np.dot(l0, synapse_sets[0].value)))
    l2 = np.array(nonlin(np.dot(l1, synapse_sets[1].value)))

    print(l2.mean())

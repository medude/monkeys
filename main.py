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
with open("input/input.json", "r") as data_file:
    data = json.loads(data_file.read())
X = np.array(data["train"]["input"])
y = np.array(data["train"]["output"])

# make it deterministic
np.random.seed(1)

# synapses
synapse_sets = []
# fill the synapse sets
for i in range(len(data["structure"]) - 1):
    synapse_sets.append(SynapseSet(prev_layer=data["structure"][i], next_layer=data["structure"][i + 1]))

# neuron layers
neuron_layers = []
# fill neuron layers
neuron_layers_length = len(data["structure"])

for i in range(neuron_layers_length):
    value = X if i == 0 else nonlin(np.dot(neuron_layers[i - 1].value, synapse_sets[i - 1].value))

    neuron_layers.append(Layer(value, neuron_num=data["structure"][i]["neuron_num"], bias=data["structure"][i]["bias"],
                               is_output=(i + 1 == neuron_layers_length)))

# training step
for j in range(data["statistics"]["train_to_steps"]):
    # for j in range(10):
    # Calculate forward through the network.
    # neuron_layers[0] = Layer(X, neuron_num=1, bias=1)
    # neuron_layers[1] = Layer(nonlin(np.dot(neuron_layers[0].value, synapse_sets[0].value)),
    #                          neuron_num=10)
    # neuron_layers[2] = Layer(nonlin(np.dot(neuron_layers[1].value, synapse_sets[1].value)), is_output=True)

    # Backpropagation of errors using the chain rule.
    neuron_layers[2].calc_error(y)
    neuron_layers[2].calc_delta()
    neuron_layers[1].calc_error(neuron_layers[2], synapse_sets[1])
    neuron_layers[1].calc_delta()

    # Sometimes print out the error
    if (j % data["statistics"]["record_error_every"]) == 0:
        print("Error: " + str(np.mean(np.abs(neuron_layers[2].error))))
        if np.mean(np.abs(neuron_layers[2].error)) <= data["statistics"]["train_to_error"]:
            break

    synapse_sets[0].set_layers(neuron_layers[0], neuron_layers[1])
    synapse_sets[1].set_layers(neuron_layers[1], neuron_layers[2])

    # update weights (no learning rate term)
    for synapse_set in synapse_sets:
        synapse_set.update_weights()

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

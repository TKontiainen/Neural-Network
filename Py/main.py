import numpy as np
from random import *
from math import *

def Sigmoid(x):
     return 1 / (1 + 1 / np.exp(x))

def MaxIndex(l):
    a = 0
    i = 0
    while i < len(l):
        if l[i] > l[a]:
            a = i
        i += 1
    return a


class Layer:
    # Initialises the number of incoming and outgoing nodes, gradients, weights and biases
    def __init__(self, nodes_in, nodes_out):
        self.nodes_in = nodes_in # Number Of Input Nodes
        self.nodes_out = nodes_out # Number Of Output Nodes

        self.gradientW = [[0 for _ in range(nodes_in)] for __ in range(nodes_out)] # Gradient Of The Cost Function For The Weights
        self.weights = [[0 for _ in range(nodes_in)] for __ in range(nodes_out)]

        self.gradientB = [0] * nodes_out # Gradient Of The Cost Function For The Biases
        self.biases = [0] * nodes_out
        
        self.InitializeRandomWeights()

    def InitializeRandomWeights(self):
        for i in range(self.nodes_out):
            for j in range(self.nodes_in):
                random_value = (random() * 2 - 1) / sqrt(self.nodes_in)
                self.weights[i][j] = random_value

    def ApplyGradients(self, learnrate):
        for node_out in range(self.nodes_out):
            self.biases[node_out] -= self.gradientB[node_out] * learnrate
            for node_in in range(self.nodes_in):
                self.weights[node_out][node_in] -= self.gradientW[node_out][node_in] * learnrate

    def CalculateOutputs(self, inputs):
        outputs = []
        for i in range(self.nodes_out): # Loop through all the output nodes
            output = self.biases[i] # Output of the current node
            for j in range(self.nodes_in): # Loop through all the input nodes
                output += inputs[j] * self.weights[i][j] # Add the weighted sum of all the input nodes to the output
            outputs.append(Sigmoid(output)) # Put the output through the activation function
            
        return outputs

class Network:
    def __init__(self, layer_sizes):
        self.layers = []
        for i in range(len(layer_sizes)-1):
            layer = Layer(layer_sizes[i], layer_sizes[i+1])
            self.layers.append(layer)

    def CalculateOutputs(self, inputs):
        for layer in self.layers:
            inputs = layer.CalculateOutputs(inputs)
        return inputs
    
    def Classify(self, inputs):
        outputs = self.CalculateOutputs(inputs)
        return MaxIndex(outputs)
    
    def NodeCost(self, activation, expected):
        return (activation - expected) ** 2
    
    def PrtDrvCostWrtActivation(self, activation, expected):
        return 2 * (activation - expected)
    
    def Cost(self, inputs, expected_outputs):
        cost = 0
        outputs = self.CalculateOutputs(inputs)
        for i in range(len(outputs)):
            cost += self.NodeCost(outputs[i], expected_outputs[i])
        return cost

    def AvgCost(self, data):
        cost = 0
        i = 0
        for inputs, expected_outputs in data.items():
            cost += self.Cost(inputs, expected_outputs)
            i += 1
        return cost / i
    
    
    def ApplyAllGradients(self, learnrate):
        for layer in self.layers:
            layer.ApplyGradients(learnrate)
    
    def Learn(self, data, learnrate):
        h = 0.0001
        original_cost = self.AvgCost(data)

        for layer in self.layers:

            # Calculate The Change In Cost For Each Weight
            for node_out in range(layer.nodes_out):
                for node_in in range(layer.nodes_in):
                    layer.weights[node_out][node_in] += h
                    change = self.AvgCost(data) - original_cost
                    layer.weights[node_out][node_in] -= h
                    layer.gradientW[node_out][node_in] = change / h

            # Calculate The Change In Cost For Each Bias
            for node_out in range(layer.nodes_out):
                layer.biases[node_out] += h
                change = self.AvgCost(data) - original_cost
                layer.biases[node_out] -= h
                layer.gradientB[node_out] = change / h

        self.ApplyAllGradients(learnrate)
    
def main():
    network = Network([2, 3, 2])

    safe = [1, 0]
    pois = [0, 1]

    # Training Data
    data = list({(5,8): pois, (8,8): pois, (4,8): pois, (7,8): pois,
        (9,7): pois, (9,6): pois, (9,5): pois, (10,5): pois,
        (10,4): pois, (10,3): pois, (10,2): pois, (10,1): pois,
        (8,9): pois, (4,10): pois, (3,10): pois, (6,7): pois,
        (6,10): pois, (8,6): pois, (4,9): pois, (8,4): pois,
        (8,2): pois, (9,2): pois, (9,3): pois, (6,6): pois,
        (3,7): pois, (2,8): pois, (1,8): pois, (6,9): pois,
        (5,6): safe, (2,6): safe, (2,4): safe, (1,1): safe,
        (2,2): safe, (3,2): safe, (5,1): safe, (6,4): safe,
        (7,2): safe, (8,1): safe, (3,6): safe, (3,5): safe,
        (5,5): safe, (4,4.5): safe, (5,4): safe, (3,3.5): safe,
        (4,3): safe, (5.5,3): safe, (6,2.5): safe, (4.5,2): safe}.items()) # Turn the data into a list in order to create mini batches
    
    shuffle(data) # Shuffle the data

    data, test_data = data[:42], dict(data[42:])

    # Create The Mini Batches
    num_batches = 7
    batch_size = len(data) // num_batches
    mini_batches = []
    for i in range(0, len(data)-batch_size+1, batch_size):
        batch = dict(data[i:i+batch_size])
        mini_batches.append(batch)

    data = dict(data) # Turn the data back into a dictionary
    
    i = 1
    cost = network.AvgCost(data)
    for _ in range(1000):
        for batch in mini_batches:
            network.Learn(batch, cost) 
        cost = network.AvgCost(data)
        print(i, round(cost, 5))
        i += 1

    if cost < 0.1:
        print('\n'.join([str(layer.weights) for layer in network.layers]))

    correct = 0

    for inputs, expected_output in data.items():
        output = network.Classify(inputs)
        if output == (0 if expected_output == safe else 1):
            correct += 1
        else:
            print(f"Input: {inputs} Output: {'pois' if output else 'safe'} Expected: {'pois' if expected_output == pois else 'safe'}")
        
    print(f"correct: {correct}/{len(data.keys())}")

    correct = 0

    for inputs, expected_output in test_data.items():
        output = network.Classify(inputs)
        if output == (0 if expected_output == safe else 1):
            correct += 1
        else:
            print(f"Input: {inputs} Output: {'pois' if output else 'safe'} Expected: {'pois' if expected_output == pois else 'safe'}")
    print(f"correct: {correct}/{len(test_data.keys())}")
    [[1.1953163156648539, -2.3537248305736327], [-0.8755153935490199, -1.0974085618575424], [-1.9926417554499483, -0.8566865622301134]]
[[6.215197218806081, 5.467199264070173, 7.373075119604682], [-6.272132846323858, -5.333448566864373, -7.423907180379741]]

if __name__ == "__main__":
    main()
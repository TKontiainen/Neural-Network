#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include "network.h"
#include "layer.h"
#include "functions.h"
#include "datapoint.h"

// Create a new neural network
network Network(int numLayers, int* layerSizes) {
    network network;

    network.numLayers = numLayers-1;
    network.layerSizes = layerSizes;
    network.layers = malloc(network.numLayers * sizeof(layer));

    network.numInputs = *layerSizes;
    network.numOutputs = *(layerSizes+network.numLayers);
    network.maxInputs = 0; // How many inputs does the layer with the most inputs take

    // Initialize the layers
    int numLayer;
    layer layer;
    int numNodesIn, numNodesOut;
    for (numLayer = 0; numLayer < network.numLayers; ++numLayer) {

        numNodesIn = *(layerSizes+numLayer); // Number of incoming values
        numNodesOut = *(layerSizes+numLayer+1);
        layer = Layer(numNodesIn, numNodesOut); // Create layer
        *(network.layers+numLayer) = layer; // Put layer into layers array'

        // update maxInputs
        if (numNodesIn > network.maxInputs) {
            network.maxInputs = numNodesIn;
        }
    }

    return network;
}

// Free all the memory used by the network
void FreeNetwork(network network) {
    layer layer;
    int numLayer;
    for (numLayer = 0; numLayer < network.numLayers; ++numLayer) {
        layer = *(network.layers+numLayer);
        FreeLayer(layer);
    }

    free(network.layers);
}

// Calculate the outputs for a network
void CalculateNetworkOutputs(network network, double* inputs, double* outputs) {

    double* temp_inputs = (double*)malloc(network.maxInputs*sizeof(double)); // Temporary inputs passed from layer to layer
    copy(inputs, temp_inputs, network.numInputs);

    layer layer;
    int numLayer;
    // Loop through the layers and pass the outputs to the next layer
    for (numLayer = 0; numLayer < network.numLayers; ++numLayer) {
        layer = *(network.layers + numLayer);
        copy(temp_inputs, layer.activations, layer.numNodesIn);
        CalculateLayerActivations(layer, temp_inputs, temp_inputs);
    }

    copy(temp_inputs, outputs, network.numOutputs);
    free(temp_inputs);
}

// Get the cost of a single output node (a-y)^2
double NodeCost(double activation, double expectedActivation) {
    double diff = activation - expectedActivation;
    return pow(diff, 2.0);
}

// Get the cost for a single datapoint
double Cost(network network, dataPoint dataPoint) {
    double cost;
    cost = 0.0;

    double* outputs = malloc(network.numOutputs*sizeof(double));
    CalculateNetworkOutputs(network, dataPoint.inputs, outputs);
    
    double output, expectedOutput;
    int i;
    for (i = 0; i < network.numOutputs; ++i) {
        output = *(outputs + i);
        expectedOutput = *(dataPoint.expectedOutputs + i);
        cost += NodeCost(output, expectedOutput);
    }

    return cost;
}

// Get the average cost for multiple datapoints
double AverageCost(network network, dataPoint* dataPoints, int numDataPoints) {
    double cost;
    dataPoint dataPoint;
    int i;

    cost = 0;
    for (i = 0; i < numDataPoints; ++i) {
        dataPoint = *(dataPoints+i);
        cost += Cost(network, dataPoint);
    }

    double avg = cost / i;

    return avg;
}

// Derivative of the cost function with respect to the activation value
double DerivativeNodeCostWrtActivation(double activation, double expectedActivation) {
	return 2.0 * (activation - expectedActivation);
}

// Apply gradients for all layers
void ApplyAllGradients(network network, double learnRate) {
    layer layer;
    int numLayer;
    for (numLayer = 0; numLayer < network.numLayers-1; ++numLayer) {
        layer = *(network.layers+numLayer);
        ApplyGradients(layer, learnRate);
    }
}

// Calculate the gradients
void CalculateGradients(network network, double derivative, int numActivation, int numLayer, int numDataPoints) {
    if (numLayer < 0) {
        return;
    }
    layer layer = *(network.layers + numLayer);
    double weightedInput = *(layer.weightedInputs + numActivation);
    derivative *= DerivativeActivationWrtWeightedInput(weightedInput);
    int nodeIn;
    double activation, weight;

    *(layer.gradientB + numActivation) = derivative / numDataPoints;

    for (nodeIn = 0; nodeIn < layer.numNodesIn; ++nodeIn) {
        activation = *(layer.activations + nodeIn);
        weight = *(layer.weights + numActivation * layer.numNodesIn + nodeIn);
        *(layer.gradientW + numActivation * layer.numNodesIn + nodeIn) = activation * derivative / numDataPoints;
        CalculateGradients(network, weight*derivative, nodeIn, numLayer-1, numDataPoints);
    }
}

// Backpropagation
void BackPropagate(network network, dataPoint* dataPoints, int numDataPoints, int learnRate) {
    double* outputs = (double*)malloc(network.numOutputs*sizeof(double)); // The output layer activations
    int numDataPoint, numOutput; 
    dataPoint dataPoint;
    double output, expectedOutput, a;

    // Loop through every datapoint
    for (numDataPoint = 0; numDataPoint < numDataPoints; ++numDataPoint) {
        // Calculate output values for datapoint
        dataPoint = *(dataPoints + numDataPoint);
        CalculateNetworkOutputs(network, dataPoint.inputs, outputs);

        // Loop through all the output nodes
        for (numOutput = 0; numOutput < network.numOutputs; ++numOutput) {
            output = *(outputs + numOutput);
            expectedOutput = *(dataPoint.expectedOutputs + numOutput);
            a = DerivativeNodeCostWrtActivation(output, expectedOutput);
            CalculateGradients(network, a, output, network.numLayers-1, numDataPoints);
        }
    }

    ApplyAllGradients(network, learnRate);
}


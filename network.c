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

    network.numLayers = numLayers;
    network.layerSizes = layerSizes;
    network.layers = malloc((numLayers) * sizeof(layer));
    network.lastLayer = *(network.layers+network.numLayers-1);

    network.maxInputs = 0; // How many inputs does the layer with the most inputs take
    network.numInputs = *layerSizes;
    network.numOutputs = *(layerSizes+numLayers);

    // Initialize the layers
    int numLayer;
    layer layer;
    int numNodesIn, numNodesOut;
    for (numLayer = 0; numLayer < numLayers; numLayer++) {
        numNodesIn = *(layerSizes+numLayer); // Get the number of incoming nodes
        numNodesOut = *(layerSizes+numLayer+1); // Get the number of outgoing nodes
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

    double* temp_inputs = (double*)malloc(network.maxInputs*sizeof(double)); // Create temporary inputs
    int numInputs = (*network.layers).numNodesIn; // Get the number of inputs for the first layer
    copy(inputs, temp_inputs, numInputs); // Copy the inputs into temporary inputs

    // Calculate the activations for hidden layers
    layer layer;
    int numLayer;
    for (numLayer = 0; numLayer < network.numLayers-1; ++numLayer) {
        layer = *(network.layers+numLayer);
        CalculateLayerActivations(layer, temp_inputs, temp_inputs);
    }

    // Calculate the output of the last layer
    CalculateLayerActivations(network.lastLayer, temp_inputs, outputs);

    free(temp_inputs);
}

double NodeCost(double activation, double expectedActivation) {
    double diff = activation - expectedActivation;
    return pow(diff, 2.0);
}

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

<<<<<<< HEAD
double DerivativeCostWrtActivation(double activation, double expectedActivation) {
    return 2.0 * (activation - expectedActivation);
}

void ApplyAllGradients(network network, double learnRate) {
    layer layer;
    int numLayer;
    for (numLayer = 0; numLayer < network.numLayers-1; ++numLayer) {
        layer = *(network.layers+numLayer);
        ApplyGradients(layer, learnRate);
    }
}
=======
double DerivativeNodeCostWrtActivation(double activation, double expectedActivation) {
	return 2.0 * (activation - expectedActivation);
}


void BackPropagate(network network, dataPoint* dataPoints, int numDataPoints, int learnrate) {

	double* outputs = (double*)malloc(network.numOutputs*sizeof(double));

	dataPoint dataPoint;

	double output, expectedOutput, derivative;

	int numDataPoint, i;
	for (numDataPoint = 0; numDataPoint < network.numOutputs; ++numDataPoint) {
		dataPoint = *(dataPoints + numDataPoint);
		CalculateNetworkOutputs(network, dataPoint.inputs, outputs);
		// Go through each output node and calculate it's cost
		for (i = 0; i < network.numOutputs; ++i) {
			output = *(outputs+i);
			expectedOutput = *(dataPoint.expectedOutputs+i);
			derivative = DerivativeNodeCostWrtActivation(output, expectedOutput);
		}
	}
}
>>>>>>> 40c6e17098670b2318a5ba75341dea69e3026097

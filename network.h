#ifndef NETWORK_H_
#define NETWORK_H_

#include "layer.h"
#include "datapoint.h"

typedef struct {
    int numLayers; // The number of layers (excl. input layer)
    int* layerSizes; // The sizes of the layers (including input layer)
    int maxInputs; // How many inputs does the layer with the most inputs take
    int numInputs; // How many inputs does the first layer take
    int numOutputs; // How many outputs does the last layer have
    layer* layers; // The layers
} network;

// Create a new neural network
network Network(int numLayers, int* layerSizes);

// Free all the memory used by the network
void FreeNetwork(network network);

// Calculate the outputs for a network
void CalculateNetworkOutputs(network network, double* inputs, double* outputs);

// Get the cost of a single output node (a-y)^2
double NodeCost(double activation, double expectedActivation);

// Get the cost for a single datapoint
double Cost(network network, dataPoint dataPoint);

// Get the average cost for multiple datapoints
double AverageCost(network network, dataPoint* dataPoints, int numDataPoints);

#endif
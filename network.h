#ifndef NETWORK_H_
#define NETWORK_H_

#include "layer.h"
#include "datapoint.h"

#define firstLayer (*network.layers)

typedef struct {
    int numLayers; // The number of layers

    int* layerSizes; // The sizes of the layers

    layer* layers; // The layers

    int numInputs; // How many inputs does the first layer take

    int numOutputs; // How many outputs does the last layer have

    int maxInputs; // How many inputs does the layer with the most inputs take


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

// Derivative of the cost function with respect to the activation value
double DerivativeNodeCostWrtActivation(double activation, double expectedActivation);

// Calculate the gradients
void CalculateGradients(network network, double derivative, int numActivation, int numLayer, int numDataPoints);

// Apply gradients for all layers
void ApplyAllGradients(network network, double learnRate);

// Backpropagation
void BackPropagate(network network, dataPoint* dataPoints, int numDataPoints, double learnrate);

#endif

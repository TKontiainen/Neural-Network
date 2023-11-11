#ifndef NETWORK_H_
#define NETWORK_H_

#include "layer.h"
#include "datapoint.h"

/*
The network structure has the following properties
    -- The number of layers
    -- The array of the layers
    -- The number of inputs (number of incoming nodes for the first layer)
    -- The number of outputs (number of outgoing nodes for the last layer)
    -- The number of incoming nodes for the layer with the most incoming nodes
    -- The array of the output values
*/
typedef struct
{

    int numLayers;
    Layer* layers;

    int numInputs;
    int numOutputs;
    int maxNodesIn;

    double* outputValues;

} Network;

#define firstLayer (network.layers[0])
#define lastLayer (network.layers[network.numLayers - 1])

// NewNetwork() returns a new network
Network NewNetwork(int numLayers, int* layerSizes);

// FreeNetwork() frees all the memory used for a network
void FreeNetwork(Network network);

/* CalculateNetworkOutput() calculates the outputs of the first layer
with the inputs and then uses those outputs to calculate the next
layers outputs and so on */
void CalculateNetworkOutputs(Network network, double* inputs);

// NodeCost() returns the cost of a single output node (a-y)^2
double NodeCost(double activation, double expectedActivation);

double DerivativeNodeCostWrtActivation(double activation, double expectedActivation);

// Cost() returns the total cost of the network for a single datapoint
double Cost(Network network, DataPoint dataPoint);

// AverageCost() returns the average cost for a number of datapoints
double AverageCost(Network network, DataPoint* dataPoints, int numDataPoints);

// ApplyAllGradients() calls ApplyGradients() for all layers
void ApplyAllGradients(Network network, double learnRate);

// ClearAllGradients() calls ClearGradients() for all layers
void ClearAllGradients(Network network);

// UpdateAllGradients() updates the gradients of all layers
void UpdateAllGradients(Network network, DataPoint dataPoint);

// Learn() loops through all the datapoints updating the gradients for each layer and then applying the gradients
void Learn(Network network, DataPoint* dataPoints, int numDataPoints, double learnRate);

#endif

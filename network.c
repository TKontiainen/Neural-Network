#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include "network.h"
#include "layer.h"
#include "functions.h"
#include "datapoint.h"

Network NewNetwork(int numLayers, int* layerSizes)
{
    Network network;

    network.numLayers = --numLayers;
    network.layers = (Layer*)malloc(numLayers * sizeof(Layer));

    network.numInputs = layerSizes[0];
    network.numOutputs = layerSizes[numLayers];
    network.maxNodesIn = network.numInputs;

    for (int layerIndex = 0; layerIndex < numLayers; layerIndex++)
    {
        int numNodesIn = layerSizes[layerIndex];
        int numNodesOut = layerSizes[layerIndex + 1];

        Layer layer = NewLayer(numNodesIn, numNodesOut);
        network.layers[layerIndex] = layer;

        if (numNodesIn < network.maxNodesIn)
        {
            network.maxNodesIn = numNodesIn;
        }
    }

    network.outputValues = (double*)malloc(network.numOutputs * sizeof(double));

    return network;
}

void FreeNetwork(Network network)
{
    // Loop trough and call FreeLayer() on each layer
    for (int layerIndex = 0; layerIndex < network.numLayers; layerIndex++)
    {
        Layer layer = network.layers[layerIndex];
        FreeLayer(layer);
    }

    free(network.layers);
    free(network.outputValues);
}

void CalculateNetworkOutputs(Network network, double* inputs)
{
    // Copy the inputs into the first layer
    dcopy(inputs, firstLayer.inputs, network.numInputs);

    // Loop through all the hidden layers and copy their outputs into the next layers inputs
    for (int layerIndex = 0; layerIndex < network.numLayers - 1; layerIndex++)
    {
        Layer layer = network.layers[layerIndex];
        CalculateLayerActivations(layer);
        
        Layer nextLayer = network.layers[layerIndex + 1];
        dcopy(layer.outputActivations, nextLayer.inputs, layer.numNodesOut);
    }

    // Calculate the outputs of the last layer
    CalculateLayerActivations(lastLayer);

    // Copy them into network.outputValues
    dcopy(lastLayer.outputActivations, network.outputValues, network.numOutputs);
}

double NodeCost(double activation, double expectedActivation) {
    double diff = activation - expectedActivation;
    return diff * diff;
}

double DerivativeNodeCostWrtActivation(double activation, double expectedActivation) {
	return 2.0 * (activation - expectedActivation);
}

double Cost(Network network, DataPoint dataPoint)
{
    double cost = 0.0;

    // Calculate the outputs of the network
    CalculateNetworkOutputs(network, dataPoint.inputs);

    // Loop through each output and add its nodecost to the total cost
    for (int outputIndex = 0; outputIndex < network.numOutputs; outputIndex++)
    {
        double output = network.outputValues[outputIndex];
        double expectedOutput = dataPoint.expectedOutputs[outputIndex];
        cost += NodeCost(output, expectedOutput);
    }

    return cost;
}

double AverageCost(Network network, DataPoint* dataPoints, int numDataPoints)
{
    double cost = 0.0;

    for (int dataPointIndex = 0; dataPointIndex < numDataPoints; dataPointIndex++) {
        DataPoint dataPoint = *(dataPoints + dataPointIndex);
        cost += Cost(network, dataPoint);
    }

    double averageCost = cost / numDataPoints;

    return averageCost;
}

void ApplyAllGradients(Network network, double learnRate)
{
    for (int layerIndex = 0; layerIndex < network.numLayers; layerIndex++)
    {
        Layer layer =*(network.layers + layerIndex);
        ApplyGradients(layer, learnRate);
    }
}

void ClearAllGradients(Network network)
{
    for (int layerIndex = 0; layerIndex < network.numLayers; layerIndex++)
    {
        Layer layer = *(network.layers + layerIndex);
        ClearGradients(layer);
    }
}

void UpdateAllGradients(Network network, DataPoint dataPoint)
{
    CalculateNetworkOutputs(network, dataPoint.inputs);

    double* nodeValues = CalculateOutputLayerNodeValues(lastLayer, dataPoint.expectedOutputs);
    UpdateGradients(lastLayer, nodeValues);

    for (int hiddenLayerIndex = network.numLayers - 2; hiddenLayerIndex >= 0; hiddenLayerIndex--)
    {
        Layer hiddenLayer = network.layers[hiddenLayerIndex];
        nodeValues = CalculateHiddenLayerNodeValues(hiddenLayer, network.layers[hiddenLayerIndex + 1], nodeValues);
        UpdateGradients(hiddenLayer, nodeValues);
    }
}

void Learn(Network network, DataPoint* trainingData, int sizeTrainingData, double learnRate)
{
    for (int dataPointIndex = 0; dataPointIndex < sizeTrainingData; dataPointIndex++)
    {
        DataPoint dataPoint = trainingData[dataPointIndex];
        UpdateAllGradients(network, dataPoint);
    }

    ApplyAllGradients(network, learnRate / sizeTrainingData);
    ClearAllGradients(network);
}
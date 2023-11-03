#include <stdlib.h>
#include <stdio.h>
#include <math.h>

#include "layer.h"
#include "functions.h"

// Print a layer's weights
void PrintWeights(layer layer) {
    int nodeIn, nodeOut;
    for (nodeOut = 0; nodeOut < layer.numNodesOut; ++nodeOut) {
        for (nodeIn = 0; nodeIn < layer.numNodesIn; ++nodeIn) {
            printf("%lf\n", *(layer.weights + nodeOut * layer.numNodesIn + nodeIn));
        }
    }
}

// Create a new layer
layer Layer(int numNodesIn, int numNodesOut) {
    layer layer;

    layer.numNodesIn = numNodesIn;
    layer.numNodesOut = numNodesOut;

    layer.gradientW = (double*)malloc(numNodesIn * numNodesOut * sizeof(double));
    layer.weights = (double*)malloc(numNodesIn * numNodesOut * sizeof(double)); 
    
    layer.gradientB = (double*)malloc(numNodesOut * sizeof(double)); 
    layer.biases = (double*)malloc(numNodesOut * sizeof(double)); 

    InitializeWeightsAndBiases(layer);

    layer.activations = (double*)malloc(numNodesIn * sizeof(double));
    layer.weightedInputs = (double*)malloc(numNodesOut * sizeof(double));

    return layer;
}

// Initialize random values to all the weights and set biases to all 0
void InitializeWeightsAndBiases(layer layer) {
    int nodeIn, nodeOut;
    double r;
    double *a;
    for (nodeOut = 0; nodeOut < layer.numNodesOut; ++nodeOut) {
        *(layer.biases + nodeOut) = 0; // Set the current bias to 0
        for (nodeIn = 0; nodeIn < layer.numNodesIn; ++nodeIn) {
            r = (randb() * 2.0 - 1.0) / sqrt(layer.numNodesIn); // Calculate a random number
            a = (layer.weights + nodeOut * layer.numNodesIn + nodeIn); // Calculate the position of the weight
            *a = r; // Set the weight to the random number
        }
    }
}

// Free all the memory used for a layer
void FreeLayer(layer layer) {
    free(layer.gradientW);
    free(layer.weights);
    free(layer.gradientB);
    free(layer.biases);
}

// Calculate the activations of a layer
void CalculateLayerActivations(layer layer, double* inputs, double* activations) {
    int nodeIn, nodeOut; 
    double weightedInput, activation, input, weight;
    for (nodeOut = 0; nodeOut < layer.numNodesOut; ++nodeOut) {
        weightedInput = *(layer.biases + nodeOut);
        for (nodeIn = 0; nodeIn < layer.numNodesIn; ++nodeIn) {
            input = *(inputs + nodeIn);
            weight = *(layer.weights + nodeOut * layer.numNodesIn + nodeIn);
            weightedInput += input * weight;
        }
        *(layer.weightedInputs + nodeOut) = weightedInput;
        activation = Sigmoid(weightedInput);
        *(activations + nodeOut) = activation;
    }
}

// Apply the gradients
void ApplyGradients(layer layer, double learnRate) {
    int nodeIn, nodeOut;
    for (nodeOut = 0; nodeOut < layer.numNodesOut; ++nodeOut) {
        *(layer.biases + nodeOut) -= *(layer.gradientB + nodeOut) * learnRate;
        for (nodeIn = 0; nodeIn < layer.numNodesIn; ++nodeIn) {
            *(layer.weights + nodeOut * layer.numNodesIn + nodeIn) -= *(layer.gradientW + nodeOut * layer.numNodesIn + nodeIn) * learnRate;
        }
    }
}

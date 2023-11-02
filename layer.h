#ifndef LAYER_H_
#define LAYER_H_

typedef struct {
    int numNodesIn;
    int numNodesOut;

    double* gradientW;
    double* weights;
    /* These arrays are basically accessed like weights[nodeOut][nodeIn] */

    double* gradientB;
    double* biases;
    /* These arrays are accessed like biases[nodeOut] */

    // For backpropagation
    double* weightedInputs;
    double* activations;
} layer;

// Print a layer's weights
void PrintWeights(layer layer);

// Create a new layer
layer Layer(int numNodesIn, int numNodesOut);

// Initialize random values to all the weights and set biases to all 0
void InitializeWeightsAndBiases(layer layer);

// Free all the memory used for a layer
void FreeLayer(layer layer);

// Calculate the activations of a layer
void CalculateLayerActivations(layer layer, double* inputs, double* activations);

// Apply the gradients
void ApplyGradients(layer layer, double learnRate);

#endif
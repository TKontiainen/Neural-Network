#ifndef LAYER_H_
#define LAYER_H_

/*
The layer structure has the following properties
    -- The number of incoming and outgoing nodes
    -- The weights and biases
    -- The Gradient vectors for the weights and biases
    -- The inputs, weightedInputs and output activation values
*/
typedef struct
{

    int numNodesIn;
    int numNodesOut;

    double* costGradientW; // Gradient vector for the weights
    double* weights;

    double* costGradientB; // Gradient vector for the biases
    double* biases;

    double* inputs;
    double* weightedInputs;
    double* outputActivations;

} Layer;

// Print a layer's weights
void PrintWeights(Layer layer);

// print a layer's biases
void PrintBiases(Layer layer);

// NewLayer() returns a new layer with randomized weights
Layer NewLayer(int numNodesIn, int numNodesOut);

// InitializeWeightsAndBiases() sets random values to all the layers weights
void InitializeWeights(Layer layer);

// FreeLayer() frees all the memory used for a layer
void FreeLayer(Layer layer);

// CalculateLayerActivations() calculates the output activations for a layer
void CalculateLayerActivations(Layer layer);

/* ApplyGradients() loops through the weights and biases and substracts
the relevant gradient value multiplied by the learn rate */
void ApplyGradients(Layer layer, double learnRate);

// ClearGradients() sets all the gradient values to 0
void ClearGradients(Layer layer);

// CalculateOutputLayerNodeValues() calculates the nodeValues for the output layer
double* CalculateOutputLayerNodeValues(Layer outputLayer, double* expectedOutputs);

// CalcualteHiddenLayerNodeValues() calculates the nodeValues for a hidden layer
double* CalculateHiddenLayerNodeValues(Layer hiddenLayer, Layer oldLayer, double* oldNodeValues);

// UpdateGradients() updates the gradients for a layer
void UpdateGradients(Layer layer, double* nodeValues);

#endif
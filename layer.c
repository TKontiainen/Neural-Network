#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include "layer.h"
#include "network.h"
#include "functions.h"

void PrintWeights(Layer layer)
{
    for (int nodeOut = 0; nodeOut < layer.numNodesOut; nodeOut++)
    {
        for (int nodeIn = 0; nodeIn < layer.numNodesIn; nodeIn++)
        {
            printf("DEBUG: weights[%d][%d] = %lf\n", nodeOut, nodeIn, layer.weights[nodeOut * layer.numNodesIn + nodeIn]);
        }
    }
}
void PrintBiases(Layer layer)
{
    for (int nodeOut = 0; nodeOut < layer.numNodesOut; nodeOut++) 
    {
        printf("DEBUG: biases[%d] = %lf\n", nodeOut, layer.biases[nodeOut]);
    }
}

Layer NewLayer(int numNodesIn, int numNodesOut)
{
    Layer layer;

    layer.numNodesIn  = numNodesIn;
    layer.numNodesOut = numNodesOut;

    layer.costGradientW = (double*)malloc(numNodesIn * numNodesOut * sizeof(double));
    layer.weights       = (double*)malloc(numNodesIn * numNodesOut * sizeof(double));

    layer.costGradientB = (double*)malloc(numNodesOut * sizeof(double)); 
    layer.biases        = (double*)malloc(numNodesOut * sizeof(double)); 
    
    InitializeWeights(layer);

    layer.inputs            = (double*)malloc(numNodesIn * sizeof(double));
    layer.weightedInputs    = (double*)malloc(numNodesOut * sizeof(double));
    layer.outputActivations = (double*)malloc(numNodesOut * sizeof(double));

    return layer;
}

void InitializeWeights(Layer layer)
{
    for (int nodeOut = 0; nodeOut < layer.numNodesOut; nodeOut++)
    {
        for (int nodeIn = 0; nodeIn < layer.numNodesIn; nodeIn++)
        {
            double randomValue = (randb(2.0) * 2.0 - 1.0);
            randomValue /= sqrt(layer.numNodesIn); // Scale the random value with the number of inputs
            layer.weights[nodeOut * layer.numNodesIn + nodeIn] = randomValue;
        }
    }
}

void FreeLayer(Layer layer)
{
    free(layer.costGradientW);
    free(layer.weights);
    free(layer.costGradientB);
    free(layer.biases);
    free(layer.inputs);
    free(layer.weightedInputs);
    free(layer.outputActivations);
}

void CalculateLayerActivations(Layer layer)
{
    for (int nodeOut = 0; nodeOut < layer.numNodesOut; nodeOut++)
    {
        double weightedInput = layer.biases[nodeOut];

        for (int nodeIn = 0; nodeIn < layer.numNodesIn; nodeIn++)
        {
            double inputNodeValue = layer.inputs[nodeIn];
            double weight = layer.weights[nodeOut * layer.numNodesIn + nodeIn];
            weightedInput += inputNodeValue * weight;
        }

        layer.weightedInputs[nodeOut] = weightedInput;

        double outputActivationValue = ActivationFunction(weightedInput);
        layer.outputActivations[nodeOut] = outputActivationValue;
    }
}

void ApplyGradients(Layer layer, double learnRate)
{
    for (int nodeOut = 0; nodeOut < layer.numNodesOut; nodeOut++)
    {
        double biasGradientValue = layer.costGradientB[nodeOut];
        layer.biases[nodeOut] -= biasGradientValue * learnRate;

        for (int nodeIn = 0; nodeIn < layer.numNodesIn; nodeIn++)
        {
            double weightGradientValue = layer.costGradientW[nodeOut * layer.numNodesIn + nodeIn];
            // printf("DEBUG: nodeOut = %d, nodeIn = %d, weightGradientValue = %lf, weightGradientValue * learnRate = %lf\n", nodeOut, nodeIn, weightGradientValue, weightGradientValue * learnRate);
            layer.weights[nodeOut * layer.numNodesIn + nodeIn] -= weightGradientValue * learnRate;
        }
    }
}

void ClearGradients(Layer layer)
{
    for (int nodeOut = 0; nodeOut < layer.numNodesOut; nodeOut++)
    {
        layer.costGradientB[nodeOut] = 0.0;
        for (int nodeIn = 0; nodeIn < layer.numNodesIn; nodeIn++)
        {
            layer.costGradientW[nodeOut * layer.numNodesIn + nodeIn] = 0.0;
        }
    }
}

double* CalculateOutputLayerNodeValues(Layer outputLayer, double* expectedOutputs)
{
    double* nodeValues = (double*)malloc(outputLayer.numNodesOut * sizeof(double));

    for (int outputIndex = 0; outputIndex < outputLayer.numNodesOut; outputIndex++)
    {
        double costDerivative = DerivativeNodeCostWrtActivation(outputLayer.outputActivations[outputIndex], expectedOutputs[outputIndex]);
        double activationDerivative = DerivativeActivationWrtWeightedInput(outputLayer.weightedInputs[outputIndex]);
        nodeValues[outputIndex] = costDerivative * activationDerivative;
    }

    return nodeValues;
}

double* CalculateHiddenLayerNodeValues(Layer hiddenLayer, Layer oldLayer, double* oldNodeValues)
{
    double* newNodeValues = (double*)malloc(hiddenLayer.numNodesOut * sizeof(double));

    for (int newNodeIndex = 0; newNodeIndex < hiddenLayer.numNodesOut; newNodeIndex++)
    {
        double newNodeValue = 0;

        for (int oldNodeIndex = 0; oldNodeIndex < oldLayer.numNodesOut; oldNodeIndex++)
        {
            double weightedInputDerivative = oldLayer.weights[oldNodeIndex * oldLayer.numNodesIn + newNodeIndex];
            newNodeValue += weightedInputDerivative * oldNodeValues[oldNodeIndex];
        }

        newNodeValue *= DerivativeActivationWrtWeightedInput(hiddenLayer.weightedInputs[newNodeIndex]);
        newNodeValues[newNodeIndex] = newNodeValue;
    }

    free(oldNodeValues);
    
    return newNodeValues;
}

void UpdateGradients(Layer layer, double* nodeValues)
{
    for (int nodeOut = 0; nodeOut < layer.numNodesOut; nodeOut++)
    {
        for (int nodeIn = 0; nodeIn < layer.numNodesIn; nodeIn++)
        {
            // Update the weight gradients
            double weightGradientValue = layer.inputs[nodeIn] * nodeValues[nodeOut];
            layer.costGradientW[nodeOut * layer.numNodesIn + nodeIn] += weightGradientValue;
        }

        // Update the bias gradients
        layer.costGradientB[nodeOut] += nodeValues[nodeOut];
    }
}

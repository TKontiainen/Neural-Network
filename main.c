// standard libraries
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

// own libraries
#include "layer.h"
#include "network.h"
#include "functions.h"
#include "datapoint.h"

int main(int argc, char** argv) {
    srand(time(0)); // Set the seed for the RNG

    int layerSizes[3] = {2, 3, 2};
    network network;
    network = Network(2, layerSizes);

    dataPoint dataPoint;
    double inputs[] = {5.0, 6.0};
    double expectedOutputs[] = {1.0, 0.0};
    dataPoint.inputs = inputs;
    dataPoint.expectedOutputs = expectedOutputs;

    double cost = Cost(network, dataPoint);
    printf("%lf\n", cost);

    FreeNetwork(network);

    return 0;
}
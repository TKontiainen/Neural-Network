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


// {{3, 3}, {4, 4}, {3, 3}, {5, 5}, {5, 5}, {9, 9}, {3, 3}, {9, 9}, {2, 2}, {2, 2}, {10, 10}, {4, 4}, {10, 10}, {3, 3}, {6, 6}, {10, 10}, {10, 10}, {8, 8}, {4, 4}, {8, 8}, {5, 5}, {1, 1}, {4, 4}, {8, 8}, {8, 8}, {2, 2}, {9, 9}, {4.5, 4.5}, {1, 1}, {8, 8}, {5, 5}, {2, 2}, {5.5, 5.5}, {3, 3}, {7, 7}, {6, 6}, {6, 6}, {5, 5}, {10, 10}, {8, 8}, {6, 6}, {3, 3}, {7, 7}, {4, 4}, {6, 6}, {9, 9}, {9, 9}, {6, 6}}
// {{0, 0}, {0, 0}, {1, 1}, {1, 1}, {1, 1}, {0, 0}, {1, 1}, {0, 0}, {1, 1}, {1, 1}, {0, 0}, {1, 1}, {0, 0}, {0, 0}, {0, 0}, {0, 0}, {0, 0}, {0, 0}, {0, 0}, {0, 0}, {1, 1}, {0, 0}, {1, 1}, {0, 0}, {0, 0}, {0, 0}, {0, 0}, {1, 1}, {1, 1}, {0, 0}, {1, 1}, {1, 1}, {1, 1}, {1, 1}, {0, 0}, {0, 0}, {0, 0}, {0, 0}, {0, 0}, {1, 1}, {1, 1}, {1, 1}, {1, 1}, {0, 0}, {1, 1}, {0, 0}, {0, 0}, {0, 0}}

int main(int argc, char** argv) {
    srand(time(0)); // Set the seed for the RNG

    int layerSizes[] = {2, 3, 3, 2};
    int numLayers = 4;

    network network;
    network = Network(numLayers, layerSizes);

    double dataInputs[48][2] = {
        {3, 3}, {4, 4}, {3, 3}, {5, 5}, {5, 5}, {9, 9}, {3, 3}, {9, 9}, {2, 2}, {2, 2}, {10, 10}, {4, 4},
        {10, 10}, {3, 3}, {6, 6}, {10, 10}, {10, 10}, {8, 8}, {4, 4}, {8, 8}, {5, 5}, {1, 1}, {4, 4}, {8, 8},
        {8, 8}, {2, 2}, {9, 9}, {4.5, 4.5}, {1, 1}, {8, 8}, {5, 5}, {2, 2}, {5.5, 5.5}, {3, 3}, {7, 7}, {6, 6},
        {6, 6}, {5, 5}, {10, 10}, {8, 8}, {6, 6}, {3, 3}, {7, 7}, {4, 4}, {6, 6}, {9, 9}, {9, 9}, {6, 6}
    };

    double dataExpectedOutputs[48][2] = {
        {0, 0}, {0, 0}, {1, 1}, {1, 1}, {1, 1}, {0, 0}, {1, 1}, {0, 0}, {1, 1}, {1, 1}, {0, 0}, {1, 1},
        {0, 0}, {0, 0}, {0, 0}, {0, 0}, {0, 0}, {0, 0}, {0, 0}, {0, 0}, {1, 1}, {0, 0}, {1, 1}, {0, 0},
        {0, 0}, {0, 0}, {0, 0}, {1, 1}, {1, 1}, {0, 0}, {1, 1}, {1, 1}, {1, 1}, {1, 1}, {0, 0}, {0, 0},
        {0, 0}, {0, 0}, {0, 0}, {1, 1}, {1, 1}, {1, 1}, {1, 1}, {0, 0}, {1, 1}, {0, 0}, {0, 0}, {0, 0}
    };

    dataPoint* dataPoints = (dataPoint*)malloc(48 * sizeof(dataPoint));
    int numDataPoint;
    for (numDataPoint = 0; numDataPoint < 48; ++numDataPoint) {
        *(dataPoints + numDataPoint) = DataPoint(*(dataInputs + numDataPoint), *(dataExpectedOutputs + numDataPoint));
    }

    double cost = AverageCost(network, dataPoints, 48);
    printf("%lf\n", cost);

    for (int i = 0; i < 1000; ++i) {
	BackPropagate(network, dataPoints, 48, 0.25);
    }
    

    cost = AverageCost(network, dataPoints, 48);
    printf("%lf\n", cost);

    FreeNetwork(network);

    return 0;
}

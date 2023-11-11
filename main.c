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

#define NUM_DATAPOINTS 48
#define BATCH_SIZE 6
#define NUM_BATCHES NUM_DATAPOINTS / BATCH_SIZE

int main(int argc, char** argv)
{
    srand(time(0)); // Set the seed for the RNG

    /***************************************************************************************************************************************/

    double dataInputs[NUM_DATAPOINTS][2] = {{5, 8}, {8, 8}, {4, 8}, {7, 8}, {9, 7}, {9, 6}, {9, 5}, {10, 5}, {10, 4}, {10, 3}, {10, 2}, {10, 1}, {8, 9}, {4, 10}, {3, 10}, {6, 7}, {6, 10}, {8, 6}, {4, 9}, {8, 4}, {8, 2}, {9, 2}, {9, 3}, {6, 6}, {3, 7}, {2, 8}, {1, 8}, {6, 9}, {5, 6}, {2, 6}, {2, 4}, {1, 1}, {2, 2}, {3, 2}, {5, 1}, {6, 4}, {7, 2}, {8, 1}, {3, 6}, {3, 5}, {5, 5}, {4, 4.5}, {5, 4}, {3, 3.5}, {4, 3}, {5.5, 3}, {6, 2.5}, {4.5, 2}};
    double dataExpectedOutputs[NUM_DATAPOINTS][2] = {{0.0, 1.0}, {0.0, 1.0}, {0.0, 1.0}, {0.0, 1.0}, {0.0, 1.0}, {0.0, 1.0}, {0.0, 1.0}, {0.0, 1.0}, {0.0, 1.0}, {0.0, 1.0}, {0.0, 1.0}, {0.0, 1.0}, {0.0, 1.0}, {0.0, 1.0}, {0.0, 1.0}, {0.0, 1.0}, {0.0, 1.0}, {0.0, 1.0}, {0.0, 1.0}, {0.0, 1.0}, {0.0, 1.0}, {0.0, 1.0}, {0.0, 1.0}, {0.0, 1.0}, {0.0, 1.0}, {0.0, 1.0}, {0.0, 1.0}, {0.0, 1.0}, {1.0, 0.0}, {1.0, 0.0}, {1.0, 0.0}, {1.0, 0.0}, {1.0, 0.0}, {1.0, 0.0}, {1.0, 0.0}, {1.0, 0.0}, {1.0, 0.0}, {1.0, 0.0}, {1.0, 0.0}, {1.0, 0.0}, {1.0, 0.0}, {1.0, 0.0}, {1.0, 0.0}, {1.0, 0.0}, {1.0, 0.0}, {1.0, 0.0}, {1.0, 0.0}, {1.0, 0.0}};
    
    DataPoint* dataPoints = (DataPoint*)malloc(NUM_DATAPOINTS * sizeof(DataPoint));
    for (int dataPointIndex = 0; dataPointIndex < NUM_DATAPOINTS; dataPointIndex++)
    {
        dataPoints[dataPointIndex] = NewDataPoint(dataInputs[dataPointIndex], dataExpectedOutputs[dataPointIndex]);
    }

    shuffle(dataPoints, NUM_DATAPOINTS);

    /***************************************************************************************************************************************/

    // Initialize the network
    int numLayers = 4;
    int layerSizes[] = {2, 16, 10, 2};
    Network network;
    network = NewNetwork(numLayers, layerSizes);

    // LEARN
    double cost = AverageCost(network, dataPoints, NUM_DATAPOINTS);

    for (int epoch = 0; epoch < 1000; epoch++)
    {
        printf("Epochs: %d, cost = %lf\n", epoch, cost);
        
        for (int batchIndex = 0; batchIndex < NUM_BATCHES; batchIndex++)
        {
            DataPoint* batch = dataPoints + batchIndex * BATCH_SIZE;
            Learn(network, batch, BATCH_SIZE, 0.25);
        }

        cost = AverageCost(network, dataPoints, NUM_DATAPOINTS);
    }

    FreeNetwork(network);
    free(dataPoints);

    return 0;
}

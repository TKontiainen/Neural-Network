#include "datapoint.h"

dataPoint DataPoint(double* inputs, double* expectedOutputs) {
    dataPoint dataPoint;

    dataPoint.inputs = inputs;
    dataPoint.expectedOutputs = expectedOutputs;

    return dataPoint;
}
#include "datapoint.h"

DataPoint NewDataPoint(double* inputs, double* expectedOutputs) {
    DataPoint dataPoint;

    dataPoint.inputs = inputs;
    dataPoint.expectedOutputs = expectedOutputs;

    return dataPoint;
}
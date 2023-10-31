#ifndef DATAPOINT_H_
#define DATAPOINT_H_

typedef struct {
    double* inputs;
    double* expectedOutputs;
} dataPoint;

// Create new datapoint
dataPoint DataPoint(double* inputs, double* expectedOutputs);

#endif
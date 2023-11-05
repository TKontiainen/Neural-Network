#ifndef DATAPOINT_H_
#define DATAPOINT_H_

typedef struct
{
    double* inputs;
    double* expectedOutputs;
} DataPoint;

// Create new datapoint
DataPoint NewDataPoint(double* inputs, double* expectedOutputs);

#endif
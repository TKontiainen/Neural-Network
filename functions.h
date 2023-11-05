#ifndef FUNCTIONS_H_
#define FUNCTIONS_H_

#include "datapoint.h"

// randb() generates a random double from 0 to 1
double randb();

// The activation function | Current: sigmoid
double ActivationFunction(double x);

// The derivative of the activation function with respect to the weighted input z
double DerivativeActivationWrtWeightedInput(double z);

// Copy double* a1 into double* a2
void dcopy(double* a1, double* a2, int n);

// Copy int* a1 into int* a2
void icopy(int* a1, int* a2, int n);

/* Arrange the N elements of ARRAY in random order.
   Only effective if N is much smaller than RAND_MAX;
   if this may not be the case, use a better random
   number generator. */
void shuffle(DataPoint* array, size_t n);

#endif
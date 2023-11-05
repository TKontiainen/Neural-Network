#include <stdlib.h>
#include <math.h>
#include "functions.h"
#include "datapoint.h"
#include "layer.h"

// randb() generates a random double from 0 to 1
double randb() {
    double r = (double)(rand() % 1000001) / 1000000.0;
    return r;
}

// The activation function | Current: sigmoid
double ActivationFunction(double x) {
    return 1.0 / (1.0 + exp(-x));
}

// The derivative of the activation function with respect to the weighted input z
double DerivativeActivationWrtWeightedInput(double z)
{
    double activation = ActivationFunction(z);
    return activation * (1 - activation);
}

void dcopy(double* a1, double* a2, int n) 
{
    int i;
    for (i = 0; i < n; ++i)
    {
        *(a2+i) = *(a1+i);
    }
}

void icopy(int* a1, int* a2, int n)
{
    int i;
    for (i = 0; i < n; ++i)
    {
        *(a2+i) = *(a1+i);
    }
}

void shuffle(DataPoint* array, size_t n)
{
    if (n > 1) {
        size_t i;
	for (i = 0; i < n - 1; i++) {
	  size_t j = i + rand() / (RAND_MAX / (n - i) + 1);
	  DataPoint t = array[j];
	  array[j] = array[i];
	  array[i] = t;
	}
    }
}
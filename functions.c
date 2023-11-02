#include <stdlib.h>
#include <math.h>
#include "functions.h"
#include "layer.h"

// randb() generates a random double from 0 to 1
double randb() {
    double r = (double)(rand() % 1000001) / 1000000.0;
    return r;
}

// The activation function | Current: sigmoid
double Sigmoid(double x) {
    return 1.0 / (1.0 + exp(-x));
}

// The derivative of the activation function with respect to the weighted input z
double DerivativeActivationWrtWeightedInput(double z) {
    double activation = Sigmoid(z);
    return activation * (1 - activation);
}

void copy(double* a1, double* a2, int n) {
    int i;
    for (i = 0; i < n; ++i) {
        *(a2+i) = *(a1+i);
    }
}
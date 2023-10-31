#ifndef FUNCTIONS_H_
#define FUNCTIONS_H_

// randb() generates a random double from 0 to 1
double randb();

// The activation function | Current: sigmoid
double Sigmoid(double x);

// The derivative of the activation function with respect to the weighted input z
double DerivativeActivationWrtWeightedInput(double z);

// Copy a1 into a2
void copy(double* a1, double* a2, int n);

#endif
#include <stdio.h>
#include <stdbool.h>
#include <stdlib.h>

// length of spikes, size of dots, is it safe? 
int data[][3] = {{5, 8, 0}, {4, 8, 0}, {5, 6, 1}, {2, 6, 1}, {2, 4, 1}};

int weight_1_1, weight_1_2, weight_2_1, weight_2_2;
int bias_1, bias_2;

bool classify(int input_1, int input_2) {
    int output_1 /*safe*/ = input_1 * weight_1_1 + input_2 * weight_1_2 + bias_1;
    int output_2 /*pois*/ = input_1 * weight_2_1 + input_2 * weight_2_2 + bias_2;

    printf("%d, %d\n", output_1, output_2);

    return output_1 > output_2;
}

bool getExpectedOutput(int input_1, int input_2) {
    int i;
    for (i = 0; i < sizeof(data); ++i) {
	if (data[i][0] == input_1 && data[i][2] == input_2) {
	    return (bool)data[i][2];
	}
    }
    return false;
}

int main(int argc, char** argv) {

    weight_1_1 = 15;
    weight_1_2 = 24;
    weight_2_1 = 13;
    weight_2_1 = 60;
    bias_1 = 15;
    bias_2 = 17;

    int input_1, input_2;

    bool output;
    bool expected_output;

    input_1 = 5;
    input_2 = 6;

    output = classify(input_1, input_2);
    expected_output = getExpectedOutput(input_1, input_2);

    char* o = output ? "safe" : "poisonus";
    char* e = expected_output ? "safe" : "poisonus";
    printf("expected: %s, actual: %s\n", e, o);

    return 0;
}

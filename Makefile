main: main.c layer.c network.c datapoint.c functions.c
	gcc main.c layer.c network.c functions.c datapoint.c -o main -lm -ggdb -O3
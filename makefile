main: main.c layer.c network.c functions.c
	gcc main.c layer.c network.c functions.c -o main -lm -ggdb
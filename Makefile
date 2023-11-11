objects = src/*.c
main: $(objects)
	gcc -o main $(objects) -lm -ggdb -O3

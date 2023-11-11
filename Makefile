objects = *.c
main: $(objects)
	gcc -o main $(objects) -lm -ggdb -O3

all:
	nvcc -x cu main.cpp -Xcompiler -fopenmp -o main.x -arch=sm_75 -std=c++11

run:
	./main.x 20

clean:
	-rm -f *.x *.nsys-rep *.sqlite

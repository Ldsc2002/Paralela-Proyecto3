all: pgm.o houghBase houghConst houghShared

houghShared: houghShared.cu pgm.o
	nvcc houghShared.cu pgm.o -ljpeg -o houghShared

houghConst: houghConst.cu pgm.o
	nvcc houghConst.cu pgm.o -ljpeg -o houghConst

houghBase: houghBase.cu pgm.o
	nvcc houghBase.cu pgm.o -ljpeg -o houghBase

pgm.o: pgm.cpp
	g++ -std=c++17 -c pgm.cpp -o pgm.o

run1:
	./houghBase runway.pgm 4000

run2:
	./houghConst runway.pgm 4000

run3:
	./houghShared runway.pgm 4000
# CellSim3D
Welcome to the CellSim3D repository. CellSim3D is a software package to simulate the mechanistic aspects of cell division in three dimensions. The code is written in C/C++ and CUDA. 

## References: 

If you use the code, please cite the following papers:

The 3D model is based on the 2D model introduced in

A new model for cell division and migration with spontaneous topology changes, Anna Mkrtchyan, J.A. Åström and Mikko Karttunen, Soft Matter 10, 4332-4339 (2014).

3D model:

CellSim3D: GPU Accelerated Software for Simulations of Cellular Growth and Division in Three Dimensions, 
P. Madhikar, J.A. Astrom, J. Westerholm, M. Karttunen

## Compilation
CellSim3D can be compiled easily with the packaged makefile. First
change the -arch variable to correspond to the compute capability of
your GPU.

For example: -arch=sm_52 for a GPU of compute capability of 5.2

Then:

mkdir bin
make -j12 CellDiv to compile the simulator.

The simulator can be found in the bin directory

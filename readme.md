# CellSim3D
Welcome to the CellSim3D repository

## Compilation
CellSim3D can be compiled easily with the packaged makefile. First
change the -arch variable to correspond to the compute capability of
your GPU.

For example: -arch=sm_52 for a GPU of compute capability of 5.2

Then:

mkdir bin
make -j12 CellDiv to compile the simulator.

The simulator can be found in the bin directory

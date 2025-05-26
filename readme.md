![Alt Text](https://github.com/SoftSimu/CellSim3D/raw/master/images/mov.gif)

# CellSim3D
Welcome to the CellSim3D repository. CellSim3D is a software package to simulate the mechanistic aspects of cell division in three dimensions. The code is written in C/C++ and CUDA.

## References:

If you use the code, please cite the following papers:

The 3D model is based on the 2D model introduced in

A new model for cell division and migration with spontaneous topology changes, A. Mkrtchyan, J.A. Åström and M. Karttunen, Soft Matter 10, 4332-4339 (2014).

3D model:

CellSim3D: GPU Accelerated Software for Simulations of Cellular Growth and Division in Three Dimensions,
P. Madhikar, J.A. Astrom, J. Westerholm, M. Karttunen, submitted

## Compilation
CellSim3D can be compiled easily with the packaged makefile. First
change the -arch variable to correspond to the compute capability of
your GPU.

For example: -arch=sm_52 for a GPU of compute capability of 5.2

Then:


```bash
make -j12 CellDiv to compile the simulator.
```

The simulator can be found in the bin directory

## Simulations

To run a simulation, edit the parameters in the `inp.json`  file to define your system configuration.

More details on the available features and input variables can be found in the [user guide](./description.md).

The command to run the simulation depends on which branch of the repository you are using:

- **Main branch** (older version):
  
  ```bash
  ./CellDiv [num_initial_cells] inp.json [device_number]
  ```
  For example:
  ```bash
  ./CellDiv 50 inp.json 0
  ```
- **multigpu branch** (New version):
  
  ```bash
   mpirun -np [num_processors] ./CellDiv [num_initial_cells] inp.json [sections_x] [sections_y] [sections_z]
  ```
  The last three arguments specify how the simulation domain is divided along the x, y, and z directions.
  The product of `sections_x` × `sections_y` × `sections_z` must equal `num_processors`. This ensures the domain is properly split across all MPI processes.


## Visualization in Blender

To visualize the results of the simulations using Blender, follow these steps:

  From the `bin` directory, copy the following files to the `scripts` folder.
   - `inp.xyz`   

  Open the `CellDiv.blend` file in Blender and adjust the camera settings as needed to capture the entire system.
  
Open a terminal and navigate to the `scripts` directory.

Execute the following command to run Blender in the background and render the results:
```bash
blender --background CellDiv.blend --python render.py -- inp.xyz
```


## Simulator Source Code Description (note: subject to change):

- **GPUBounce.cu**

   This file contains the entry point for the simulator code. This is
   where the GPU is selected and memory is allocated. Simulation
   parameters are read from the input json file.

   All GPU functions (force calculation, integration, cell division) is
   controlled from here.

- **propagate.cu**

   This file contains GPU kernel code that is used for force
   calculations and integration.

- **centermass.cu**

   This file only contains a single GPU kernel that calculate the
   centers of mass of individual cells. This is needed for the
   calculation of cell volumes and the cell division algorithm

 - **BondKernels.cu**

   This file calculates the equilibrium bond lengths of bonded nodes in
   a cell. For now, this code is not very crucial since this bond
   length does not change over the course of a simulation. It will be
   used in later releases more extensively.

 - **postscriptinit.cu**

   This file contains some legacy code that is no longer used. It also
   contains the implementation of the cell division algorithm.

- **propagatebound.cu**

   Code here is used to calculate the bounding boxes around cells,
   which are then used during neighbor list generation. Neighbor list
   generation itself is also carried out here.

- **PressureKernels.cu**

   Cell internal pressure is managed with the code here.

- **IntegrationKernels.cu** and **AdaptiveTimeKernels.cu**

   Code here is not currently used.

## Analysis Scripts (In the scripts/ directory):
The scripts directory contains various analysis scripts. Their
functionality can be explored by running

~~~bash
  python3 script.py --help
~~~

The most important ones are:

##### celldiv.py

This is the interface to the binary format of the trajectory. It can
be used to read the trajectory into numpy arrays.

##### render.py

This is the rendering engine used to visualize the simulations. It
produces images which can be made into movies. Requires Blender 2.7 or
higher. Uses the scene set in CellDiv.blend

Learn more by running

~~~bash
  blender -b CellDiv.blend -P scripts/render.py -- --help
~~~

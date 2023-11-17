# CellSim3D Documentation

Welcome to the documentation of CellSim3D. 
Implemented features include:

+ Symmetric and asymmetric cell division
+ Cellular apoptosis
+ Extracellular Matrix
+ Growth of heterogenous or homogenous tissue
+ Flexible boundary conditions
+ Simulation of colloids with two or less components


Features in the works:
+ Cell Polarity



# input file
Let's start with a description of the variables in the input file.

## Core variables

+ "MaxNoofC180s": Maximum number of cells we might have in the simulation (To construct arrays).
+ "div_time_steps": simulation time in which we have division #check<---
+ "non_div_time_steps": simulation time in which system progresses with no division #check <----
+ "trajWriteInt": Growth count interval, the interval in which details of the simulation are recorded
+ "binaryOutput": Write binary trajectory file -  0 for false & 1 for true
+ "time_interval": time step of simulation
+ "trajFileName": Name of the coordinates file (eg. "inp.xyz")
+ "forces_file": Force file name (eg."inp.csv")
+ "write_vel_file": Write velocity file - 0 for false & 1 for true
+ "write_for_file": Write force file - 0 for false & 1 for true
+ "write_traj_file": Write trajectory file - 0 for false & 1 for true
+ "write_cm_file": Write the position of the center of mass of the cells to file - 0 for false & 1 for true
+ "write_vcm_file": Write the center of mass velocities to file - 0 for false & 1 for true
+ "write_fcm_file": Write the center of mass forces to file - 0 for false & 1 for true
+ "correct_com": Set the center of mass of the system to zero - 0 for false & 1 for true 
+ "correct_Vcom": Set the velocity of the center of mass of the system to zero - 0 for false & 1 for true

Parallelization Parameters
+ "MaxBuffer": Buffersize
+ "MaxNeighList":


## Geometry

The geometry used to describe the cells is that of a C180 fullerene, a structure that gives our cells a spherical shape while keeping the simulation time reasonable. The input variables used to describe each cell include: 

+ "particle_mass": mass of each node constructing a cell
+ "checkSphericity": Check whether or not cells are spherical to determine if they are suitable for division,  0 for false & 1 for true
+ 

## Force field
- [ ] Would it be better to classify based on bonded/non-bonded or intracellular/Intercellular forces? 
### Intracellular forces 

#### Modelling cell membrane
We model the visco-elasticity of cell membranes with damped harmonic oscillators.
```math
F_i^B=\Sigma^{3}_{j=1} k^{B} \hat{b}_{ij}(R_{ij} - R_0) - \gamma_{int} v_{ij}
```

The input variables used to describe This force accounting for the bonded interactions between each node and its three neighboring nodes are:


+ "stiffFactor1" and "Youngs_mod" whose product is the intercellular bond stiffness $k^{B}$.
+ "internal_damping": Bond damping coefficient $\gamma_{int}$.


#### Sphericity
To keep the spherical shape of the cell, we introduce the angle force, $F_{angle}$, described by 
```math
F^{\theta}=-\nabla \left( \frac{1}{2} k_{\theta} (\theta - \theta_0)^2  \right)
```



#### Growth
The internal pressure driving cell growth can be described using two variables:
```math
F^P= P S \hat{n}
```
Where P is the internal pressure of the cell, and S is the unit element of the surface area.
+ "growth_rate": Product of S and P, Pressure growth rate.
+ "minPressure": Initial pressure force
+ "maxPressure": Final pressure force


### Intercellular forces

#### Adhesion
+ "attraction_range": Attraction range between two neighboring cells
+ We have "attraction_strength" and "Youngs_mod" such that the product of these two is the attraction stiffness.


#### Repulsion
+ "repulsion_range": Repulsion range between two neighboring cells
+ We have "repulsion_strength" and "Youngs_mod" such that the product of these two is the repulsion stiffness.


#### Friction between cells


```math
F^{F,e}_{ij}= -\gamma_{ext} v_{ij}^{\tau}
```
+ "viscotic_damping": Inter-membrane friction $\gamma_{ext}$


#### Friction between cells and medium
```math
F^{F,m} = -\gamma_m v
```

+ "gamma_visc": Medium friction $\gamma_m$

## Division 
We assume that cells divide symmetrically through their centers of mass and asymmetrically by placing the division plane off-center.

+ "division_Vol": Volume threshold for division
+ 

## ECM




## Apoptosis

In the apoptosis process, cells lose their internal pressure and shrink. This shrinkage is modeled by changing the growth rate
from positive to negative. After the volume of these cells passes a certain threshold, the cells are removed from the
system.

+ "apoptosis": (0,1) Apoptosis True/False
+ "popToStartApo": Number of cells we should have in the system to start apoptosis
+ "Apo_ratio": Probability of apoptosis for each individual cell
+ "squeeze_rate": rate of shrinkage of a cell in apoptosis (replaces growth rate)
+ "apoptosis_Vol": Cells are removed from the system after reaching this threshold volume

### Boundary
+ "shear_rate": shear rate for Lees Edwards boundary condition (LEbc), #Vshift = shear_rate*boxMax.x ?


## To add

- [ ] Write a description for all input variables
- [ ] Can include an animation & description for each feature (can change the color of cells you would like to point out later, eg. those going through apoptosis or polar ones, can have different colors for colloids that have more than one component.))
- [ ] Add the force equations
- [ ] Write better descriptions for each force




More details on this package can be found at:
1. https://ir.lib.uwo.ca/etd/9239/

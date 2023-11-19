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
+ "Restart": Allows changing of boundary conditions. After repositioning the walls and all cells in the previous system, the new simulation will continue where the previous one was left.
+ "doAdaptive_dt": Changes the time steps used in the integration- cant find where it was actually implemented? #check <----
+ "dt_max": Maximum time step used
+ "dt_tol": Minimum time step used
+ "phase_count": Updates Youngs Modules after "phase_count" steps #check <---
+ "write_cont_force": Write contact forces (attraction and repulsion) to inp.CSV file.

Counting Cells:
+ "countcells": Write a File to keep track of mitosis - 0 for false & 1 for true
+ "mit-index_file_name": Specify the name of the file (e.g. "inp.dat")
+ "count_only_internal_cells?": Count cells within a certain interval- 0 for false & 1 for true
+ "radius_cutoff": Cutoff radius of internal cells
+ "overwrite_mit_ind_file?": 0 for false & 1 for true
+ "cell_count_int": The interval in which cells are counted 



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
+ "constrainAngles": Calculates the equilibrium angle for the specific input structure- 0 for false & 1 for true



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
+ "useDivPlaneBasis": use the specified division plane - 0 for false & 1 for true
+ "divPlaneBasisX": set the division plane normal to the X direction if "useDivPlaneBasis" is set to one
+ "divPlaneBasisY": set the division plane normal to the Y direction if "useDivPlaneBasis" is set to one
+ "divPlaneBasisZ": set the division plane normal to the Z direction if "useDivPlaneBasis" is set to one
+ "asymDivision": Do assymetric division - 0 for false & 1 for true

### New cell characteristics

+ "useDifferentCell": Set different characteristics for new cells -  0 for false & 1 for true
+ "SizeFactor": 0.75,
+ "StiffFactor": 0.7,
+ "GrowthRate": growth rate of the new cell
+ "divisionV": division volume of the new cell
+ "gammaV": 0.01,
+ "VisDamping" : 0.01,
+ "Apo_rate2": Probability of apoptosis for each new cell
+ "squeeze_rate2": rate of shrinkage of the new cell in apoptosis (replaces growth rate)
+ "numberOfCells": 0,
+ "fractionOfCells": fraction of new cells that will have these properties
+ "closenessToCenter": 0.0,
+ "Yseparation": 0.0,
+ "chooseRandomCellindices": 0,
+ "daughtSame": 1,
+ "duringGrowth": 1,
+ "recalc_r0": 0


### Population regulation

+ "doPopModel": 0,
+ "totalFood": 5000.0,
+ "regular_consumption": 2.0,
+ "division_consumption": 10.0,
+ "death_release_food": 10.0,
+ "haylimit": 10,
+ "cellLifeTime": 20000




## ECM

+ "ECM": Have Extracellular Matrix in simulation - 0 for false & 1 for true
+ "Max_ECM_nodes": Maximum number of nodes in the ECM #can't find,ask <---
+ "Div_size": 1.0, #can't find
+ "Buffer_size": 1000,
+ "MaxNeighList": 512, 
+ "mass": 0.04,
+ "stiffness_min": Minimum stifness of bonds in ECM #stifness varies across ECM
+ "stiffness_max": Maximum stifness of bonds in ECM
+ "angleConstant": 1.0,
+ "vis_damp": 0.01,
+ "gamma_env": 0.1,
+ "vis_ecm_cell": 0.05,
+ "attraction_range_ecm": 0.2,
+ "repulsion_range_ecm" : 0.1,
+ "attraction_strength_ecm": 0.4,
+ "repulsion_strength_ecm" : 100,
+ "write_traj_Ecm_file": 0 for false & 1 for true




## Apoptosis

In the apoptosis process, cells lose their internal pressure and shrink. This shrinkage is modeled by changing the growth rate
from positive to negative. After the volume of these cells passes a certain threshold, the cells are removed from the
system.

+ "apoptosis": (0,1) Apoptosis True/False
+ "popToStartApo": Number of cells we should have in the system to start apoptosis
+ "Apo_ratio": Probability of apoptosis for each individual cell
+ "squeeze_rate": rate of shrinkage of a cell in apoptosis (replaces growth rate)
+ "apoptosis_Vol": Cells are removed from the system after reaching this threshold volume

## Polarity 
+ "Polarity": Not successful yet, CellStressTensor needs to be calculated using a new algorithm.

## Boundary
+ "shear_rate": shear rate for Lees Edwards boundary condition (LEbc), #Vshift = shear_rate*boxMax.x ?
+ "useRigidSimulationBox": Use rigid boundary condition, 0 for false & 1 for true
+ "usePBCs": Use Periodic Boundary Condition - 0 for false & 1 for true
+ "useLEbc": Use Lees Edwards boundary condition (LEbc) - 0 for false & 1 for true
+ "useRigidBoxZ": have Rigid walls in Z direction - 0 for false & 1 for true
+ "useRigidBoxY": have Rigid walls in Y direction - 0 for false & 1 for true
+ "useRigidBoxX": have Rigid walls in X direction - 0 for false & 1 for true

 allCMsPin[i].x < 1.5 #?

+ "threshDist": 0.01,
+ "box_len_x": Maximum position of the box in the x direction
+ "box_len_y": Maximum position of the box in the x direction
+ "box_len_z": Maximum position of the box in the x direction
+ "BoxMin_x": Minimum position of the box in the x direction (usually set to 0.0)
+ "BoxMin_y": Minimum position of the box in the y direction (usually set to 0.0)
+ "BoxMin_z": Minimum position of the box in the z-direction (usually set to 0.0)
+ "flatbox": Position the z position of all cells in the middle of the box - 0 for false & 1 for true
+ "dom_len": 1.5 #Ask <-----
+ "rand_pos": gives the initial cells random positions - 0 for false & 1 for true
+ "impurity": 0 for false & 1 for true , #Ask<---
+ "impurityNum": Number of #? Cells #Ask
+ "line": 0 for false & 1 for true
+ "LineCenter": If "line" is set to one, It sets the x position of the COM of cells to the center of the box #check <---
+ "plane": Positions all cells on a plane with fixed Z - 0 for false & 1 for true

~~~
If (colliodal_dynamics) {
  If (ReadInitialConf) {}
  else  {#sets random position} 
}

else {
  If randompositon{} #Impurity ?  #Ask
  If line{}
  If plane{}
}
~~~

## Colloidal Dynamics

+ "colloidal_dynamics": 1,
+ "dispersity": 0,
+ "dispersity_max": 1,
+ "dispersity_min": 0.7,
+ "rand_vel": 0,
+ "Two_Components": 1,
+ "SizeFactor": 0.9,
+ "Friction": 0.1,
+ "Fraction": 0.3,    	
+ "Xratio": 1.0,
+ "Yratio": 1.0,
+ "Zratio": 1.0,
+ "RandInitDir": 0,
+ "ReadInitialConf" : 0,
+ "Compressor": 0,
+ "Compress_Value_X": 0.000025,
+ "Compress_Value_Y": 0.000025,
+ "Compress_Value_Z": 0.000025

## To add

- [ ] Write a description for all input variables
- [ ] Can include an animation & description for each feature (can change the color of cells you would like to point out later, eg. those going through apoptosis or polar ones, can have different colors for colloids that have more than one component.))
- [ ] Add the force equations
- [ ] Write better descriptions for each force




More details on this package can be found at:
1. https://ir.lib.uwo.ca/etd/9239/

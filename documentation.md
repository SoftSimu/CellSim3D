# CellSim3D Documentation

Welcome to the documentation of CellSim3D. 
Implemented features include:

+ Symmetric and asymmetric cell division
+ Cellular apoptosis
+ Growth of heterogenous or homogenous tissue
+ Flexible boundary conditions
+ Extracellular Matrix
+ Simulation of colloids with two or less components


Features in the works:
+ Cell Polarity



# input file
Let's start with a description of the variables in the input file.

## Core variables

+ "MaxNoofC180s"

   Maximum number of cells we might have in the simulation (To construct arrays).
+ "div_time_steps"

   simulation time in which we have division
+ "non_div_time_steps"

   After "div_time_steps" number of steps, simulation continues with no division for "non_div_time_steps"
+ "trajWriteInt"

   Growth count interval, the interval in which details of the simulation are recorded
  
+ "binaryOutput":: Boolean

   Write binary trajectory file 
+ "time_interval"

  time step of simulation (eg. $t=0.0001$)
+ "trajFileName"

   Name of the coordinates file (eg. "inp.xyz")
+ "forces_file"
  
   Force file name (eg."inp.csv")
+ "write_vel_file"::Boolean

   Write velocity file 
+ "write_for_file"::Boolean

   Write force file 
+ "write_traj_file"::Boolean

   Write trajectory file 
+ "write_cm_file"::Boolean

   Write the position of the center of mass of the cells to file 
+ "write_vcm_file"::Boolean

   Write the center of mass velocities to file 
+ "write_fcm_file"::Boolean

   Write the center of mass forces to file 
+ "correct_com"::Boolean

  Set the center of mass of the system to zero 
+ "correct_Vcom"::Boolean

   Set the velocity of the center of mass of the system to zero 
+ "Restart"::Boolean

   Allows changing of boundary conditions. After repositioning the walls and reading cell data from the previous system, the new simulation will continue where the previous one was left.
+ "doAdaptive_dt":

   Changes the time steps used in the integration
  + "dt_max":

     Maximum time step used
  + "dt_tol": #check?

     Minimum time step used
+ "phase_count":

   Updates Youngs Modules after "phase_count" steps #check <---
+ "write_cont_force":

   Write contact forces (attraction and repulsion) to the inp.CSV file.

#### Counting Cells:
+ "countcells"::Boolean

   Write a File to keep track of mitosis
+ "mit-index_file_name":

   Specify the name of the file (e.g. "inp.dat")
+ "count_only_internal_cells?"::Boolean
  
   Count cells within a certain interval
+ "radius_cutoff":

  Cutoff radius of internal cells
+ "overwrite_mit_ind_file?"::Boolean
+ "cell_count_int":

   The interval in which cells are counted 



#### Parallelization Parameters
+ "MaxBuffer": Buffersize
+ "MaxNeighList":


## Geometry

The geometry used to describe the cells is that of a C180 fullerene, a structure that gives our cells a spherical shape while keeping the simulation time reasonable. The input variables used to describe each cell include: 

+ "particle_mass"

   mass of each node constructing a cell
+ "checkSphericity"::Boolean

   Check whether or not cells are spherical to determine if they are suitable for division, 
  

## Force field

### Intracellular forces 

#### Modelling cell membrane
We model the visco-elasticity of cell membranes with damped harmonic oscillators.
```math
F_i^B=\sum^{3}_{j=1} k^{B} \hat{b}_{ij}(R_{ij} - R_0) - \gamma_{int} v_{ij}
```
where $R_0$ is the equilibrium bond length, $v_{ij}$ is the relative velocity between nodes i and j, and $\hat{b}_{ij}$ is a vector from node i to j. The parameter $\gamma _{int}$ is the coefficient of friction that dampens the oscillations of the bonds, and the bonding spring constant is  $k^{B}$. This force accounts for the bonded interactions between each node and its three neighboring nodes.

The input variables used to describe this force are:

+ "stiffFactor1" and "Youngs_mod" whose product is the intercellular bond stiffness $k^{B}$.
+ "internal_damping"

  Bond damping coefficient $\gamma_{int}$.


#### Cell Curvature
The angle force, $F_{angle}$, is a harmonic force exerted on three neighboring nodes which opposes the deformation of the angle between these three nodes and maintains cell curvature. It is defined as
```math
F_i^{\theta}=-\frac{1}{2}  \sum_j \sum_{k\neq i}  k_{\theta} \nabla (\theta_{i j k} - \theta_0)^2
```
+ "constrainAngles"::Boolean

  Calculates the equilibrium angle for the specific input structure



#### Growth
The internal pressure driving cell growth can be described using two variables:
```math
F^P= P S \hat{n}
```
Where P is the internal pressure of the cell, and S is the unit element of the surface area.
+ "growth_rate"

  Product of S and P, Pressure growth rate.
+ "minPressure"

  Initial pressure force
+ "maxPressure"

  Final pressure force


### Intercellular forces

#### Adhesion
When two cell membranes come into contact, the adhesive component holds them together. $F_A$, the attractive force between nodes in different cells is defined as


```math
\mathbf{F}_{i, m}^{\mathrm{A}}=\sum_n \sum_j \begin{cases}\mathrm{k}^{\mathrm{A}}\left(\mathrm{r}_{i j, n}-R_0^{\mathrm{A}}\right) \hat{\mathbf{r}}_{i j, n} & \text { if } r_{i j, n}< R_0^{A} ; \\ 0 & \text { if } r_{i j, n} \geq R_0^{\mathrm{A}},\end{cases}
```

+ "attraction_range"

  Attraction range between two neighboring cells
+ We have "attraction_strength" and "Youngs_mod" such that the product of these two is the attraction stiffness.


#### Repulsion
When two cell membranes come into contact, the repulsive force keeps them apart. $F_R$, the repulsive force between nodes in different cells is defined as
```math
\begin{gathered}\mathbf{F}_{i, m}^{\mathrm{R}}=\sum_n \sum_j \begin{cases}-\mathrm{k}^{\mathrm{R}}\left(r_{i j, n}-R_0^{\mathrm{R}}\right) \hat{\mathbf{r}}_{i j, n} & \text { if } r_{i j, n}< R_0^{\mathrm{R}} ; \\ 0 & \text { if } r_{i j, n} \geq R_0^{\mathrm{R}},\end{cases} \end{gathered}
```


+ "repulsion_range"

  Repulsion range between two neighboring cells
+ We have "repulsion_strength" and "Youngs_mod" such that the product of these two is the repulsion stiffness.


#### Friction between cells
$F^F$ is the friction term separated into viscous drag due to cell-extracellular matrix interactions and intermembrane friction, which is proportional to the relative velocity tangential to the cell surface.

```math
F^{F,e}_{ij}= -\gamma_{ext} v_{ij}^{\tau}
```
+ "viscotic_damping"

  Inter-membrane friction $\gamma_{ext}$


#### Friction between cells and medium
The viscous drag due to the medium is defined as

```math
F^{F,m} = -\gamma_m v
```
+ "gamma_visc"

  Medium friction $\gamma_m$



## Division 
We assume that cells divide symmetrically through their centers of mass and asymmetrically by placing the division plane off-center.

+ "division_Vol"

   Volume threshold for division
+ "useDivPlaneBasis"::Boolean

   use the specified division plane 
+ "divPlaneBasisX"

  set the division plane normal to the X direction if "useDivPlaneBasis" is set to one
+ "divPlaneBasisY"

  set the division plane normal to the Y direction if "useDivPlaneBasis" is set to one
+ "divPlaneBasisZ"

  set the division plane normal to the Z direction if "useDivPlaneBasis" is set to one
+ "asymDivision"::Boolean

  Do asymmetric division 


### New cell characteristics

+ "useDifferentCell"::Boolean
  
  Have two different types of cells in the simulation, sets different characteristics for new cells 
+ "SizeFactor"::Boolean

  Ratio of the size of the new type of cell compared to the old one
+ "StiffFactor"

   Ratio of the stiffness of the bonds of the new type of cell compared to the old one
+ "GrowthRate"

  The growth rate of the new cell
+ "division"

  division volume of the new cell
+ "gammaV" #check

  Medium Friction for the new cell
+ "VisDamping" 
  
  Intermembrane friction
+ "Apo_rate2"

  Probability of apoptosis for each new cell
+ "squeeze_rate2"

  rate of shrinkage of the new cell in apoptosis (replaces growth rate)
+ "numberOfCells": numberofCells=roundf(fractionofCells*No_of_C180s), whats the point of having it in the code?

  The number of this second type of cells in the system.
+ "fractionOfCells"

  fraction of new cells that will have these properties
+ "closenessToCenter"

+ "numberofCells"

  cells will be turned into the second type in this specific radius.
+ "Yseparation"

   Apply changes for cells in range y $\in$ Yseperation*boxMax.y
+ "chooseRandomCellindices"::Boolean  #check

  Choose random cells to be of the new type
+ "daughtSame":: Boolean
  
+ "duringGrowth"::Boolean
  
  Does not change the property of cells if the system has growth 
+ "recalc_r0":: Boolean #Ask

  If true, does not divide the cell into a fullerene


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
+ "Max_ECM_nodes": Maximum number of nodes in the ECM, for constructing an array #can't find, ask <---
+ "Div_size": 1.0, #can't find
+ "Buffer_size": 1000,
+ "MaxNeighList": 512, 
+ "mass": 0.04,
+ "stiffness_min": Minimum stiffness of bonds in ECM #stifness varies across ECM
+ "stiffness_max": Maximum stiffness of bonds in ECM
+ "angleConstant": 1.0,
+ "vis_damp": 0.01,
+ "gamma_env": 0.1,
+ "vis_ecm_cell": 0.05,
+ "attraction_range_ecm": 0.2,
+ "repulsion_range_ecm": 0.1,
+ "attraction_strength_ecm": 0.4,
+ "repulsion_strength_ecm": 100,
+ "write_traj_Ecm_file": 0 for false & 1 for true




## Apoptosis

In the apoptosis process, cells lose their internal pressure and shrink. This shrinkage is modeled by changing the growth rate
from positive to negative. After the volume of these cells passes a certain threshold, the cells are removed from the
system.

+ "apoptosis": (0,1) Apoptosis True/False
+ "popToStartApo": Number of cells we should have in the system to start apoptosis
+ "Apo_ratio": Probability of apoptosis for each cell
+ "squeeze_rate": rate of shrinkage of a cell in apoptosis (replaces growth rate)
+ "apoptosis_Vol": Cells are removed from the system after reaching this threshold volume


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
  If randompositon{} #Impurity?  #Ask
  If line{}
  If plane{}
}
~~~
## Random number Generator
+ "add_rands": 1,
+ "rand_seed": -1, // use time if < 0
+ "rand_dist": 0, # 0 - uniform (default) / Others to be implemented
+ "rand_scale_factor": 0.1

## To add

- [ ] Write a description for all input variables
- [ ] Can include an animation & description for each feature (can change the color of cells you would like to point out later, eg. those going through apoptosis or polar ones, can have different colors for colloids that have more than one component.))
- [ ] Add the force equations
- [ ] Write better descriptions for each force




More details on this package can be found at:
1. https://ir.lib.uwo.ca/etd/9239/

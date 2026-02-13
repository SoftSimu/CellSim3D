# Getting started

<font size="3"> 

CellSim3D is a 3D cell-based simulation software for modeling cell and tissue mechanics. This guide will walk you through the steps to [compile](#compilation) the software, [run simulations](#Simulations), and [visualize](#visualization-in-blender) the results. The [input parameters](#input-file-structure) for the simulations are organized into different groups, which are explained in detail below.
</font>


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
<br><br> 

## Simulations

To run a simulation, edit the parameters in the `inp.json`  file to define your system configuration. 


```bash
  mpirun -np [num_processors] ./CellDiv [num_initial_cells] inp.json [sections_x] [sections_y] [sections_z]
```

For example:
```bash
mpirun -np 1 ./CellDiv 100 inp.json 1 1 1
```
The last three arguments specify how the simulation domain is divided along the x, y, and z directions.
The product of `sections_x` × `sections_y` × `sections_z` must equal `num_processors`. This ensures the domain is properly split across all MPI processes.

<br> <br> 

# Input File Structure

The simulation input file is written in JSON format and organized into
the following parameter groups:

<font size="4">

**[1. Core Simulation values](#1-simulation_core)**\
**[2. Cell physics](#2-cell_physics)**\
**[3. Apoptosis](#3-apoptosis)**\
**[4. Extracellular Matrix (ECM)](#4-extracellular-matrix-ecm)**\
**[5. Division](#5-division)**\
**[6. New cell characteristics](#6-new-cell-characteristics)**\
**[7. Boundary conditions and simulation box specifications](#7-boundary-conditions-and-simulation-box-specifications)**\
**[8. Random number Generator](#8-random-number-generator)**\
**[9. Colloidal Dynamics](#9-colloidal-dynamics)**\
**[10. Fluid forces](#10-fluid-forces)**\
**[11. Counting Cells (legacy)](#11-counting-cells-legacy)**\
**[12. Population control (legacy)](#12-population-control-legacy)**\
**[13. System shape](#13-system-shape)**

</font>

# 1. simulation_core

## Time, System Size and Control


<div align="center">

| Parameter | Type | Description |
|-----------|------|------------|
| `Max_cell_count` | int | Maximum number of cells allowed in simulation. Used for memory allocation |
| `div_time_steps` | int | Number of time steps with active division. |
| `non_div_time_steps` | int | Additional steps without division. |
| `dt` | float | Simulation time step. (eg. $dt=0.0001$) |
| `Restart_simulation` | bool | Restart the simulation from the last step of the previous simulation|
| `correct_com` | bool | Set the center of mass of the system to zero |
| `correct_Vcom` | bool | Set the velocity of the center of mass of the system to zero |
| `MaxNeighborList_size` | int | Maximum number of neighbors in the neighbor list. Used for memory allocation |
| `MaxBuffer` | int | Buffer size for neighbor list construction. Used for memory allocation

</div>


## Output Control



<div align="center">

| Parameter | Type | Description |
|-----------|------|------------|
| `trajFileName` | string | Name of the trajectory file (eg. "inp.xyz") |
| `trajWriteInt` | int | The interval in which the outputs are written in the file. |
| `binaryOutput` | bool | Write binary trajectory file |
| `write_traj_file` | bool | Write the trajectory output file or not |
| `write_cont_force` | bool | Write contact forces to the inp.CSV file. |
| `forces_file` | string | Force file name (eg."inp.csv") |
| `write_vel_file` | bool | Write the velocity output file or not |
| `write_for_file` | bool | Write the force output file or not |
| `write_cm_file` | bool | Write the position of the center of mass of the cells to file |
| `write_vcm_file` | bool | Write the center of mass velocities to file |
| `write_fcm_file` | bool | Write the center of mass forces to file |

</div>


# 2. cell_physics

Each cell in the simulation is made of 180 nodes, and each node and its neighbors are connected with springs in the shape of a C180 fullene:


<div align="center">

  ![Scheme_Cell](https://github.com/SoftSimu/CellSim3D/tree/multigpu_2025/images/Cell_ShapeInd4.89.png)

</div>



## Intracellular forces: modeling cell structure

#### **Cell membrane and cytoskeleton**
We model the visco-elasticity of cell membranes with damped harmonic oscillators.
```math
F_i^B=\sum^{3}_{j=1} k^{B} \hat{b}_{ij}(R_{ij} - R_0) - \gamma_{int} v_{ij}
```
where $R_0$ is the equilibrium bond length, $v_{ij}$ is the relative velocity between nodes i and j on a cell, and $\hat{b}_{ij}$ is a vector from node i to j on a cell. The parameter $\gamma _{int}$ is the coefficient of friction that dampens the oscillations of the bonds, and the bonding spring constant is  $k^{B}$. This force accounts for the bonded interactions between each node and its three nearest neighboring nodes.




#### **Cell Curvature**
The angle force, $F_{angle}$, is a harmonic force exerted on three neighboring nodes which opposes the deformation of the angle among them and maintains cell curvature. It is defined as
```math
F_i^{\theta}=-\frac{1}{2}  \sum_j \sum_{k\neq i}  k_{\theta} \nabla (\theta_{i j k} - \theta_0)^2
```
#### **Growth**
The internal pressure driving cell growth can be described using P, the internal pressure of the cell, and S, the unit element of the surface area.
```math
F^P= P S \hat{n}
```
#### **Medium friction**
The viscous drag on a cell due to the surrounding medium is applied as

```math
F^{F,m} = -\gamma_m v
```

 <br> <br> 

<div align="center">

| Parameter | Symbol | Type | Description |
|---|---|---|---|
| `particle_mass` | $m$ | float | Mass of each node on the cell |
| `Youngs_modulus` | $E$ | float | Young's modulus |
| `StiffFactor` | $SF$ | float | Stiffness factor ($SF \times E = k_B$ intracellular bond stiffness) |
| `Internal_damping` | $\gamma_{int}$ | float | Damping coefficient for the intracellular bonds |
| `minPressure` | $P_{min}$ | float | Minimum internal pressure of the cell |
| `maxPressure` | $P_{max}$ | float | Maximum internal pressure of the cell |
| `growth_rate` | $\Delta P$ | float | Rate at which the internal pressure increases |
| `division_volume` | $V_{div}$ | float | Volume threshold for cell division |
| `constrainAngles` | | bool | Determines whether angle forces are applied to maintain cell curvature |
| `angle_Constant` | $k_\theta$ | float | Stiffness constant for the angle forces |
| `medium_viscosity` | $\gamma_m$ | float | Viscosity of the surrounding medium, affecting friction between cells and medium |
| `checkSphericity` | | bool | Determines whether to check for sphericity condition for cell division |
| `Polarity`| | bool | Determines whether to include cell polarity in the simulation |
| `gravity` | $g$ | float | Gravitational force applied to the cells |


</div>


## Intercellular forces

#### Adhesion
When two cell membranes come into contact, the adhesive component holds them together. $F_A$, the attractive force between nodes on different cells is defined as


```math
\mathbf{F}_{i, m}^{\mathrm{A}}=\sum_n \sum_j \begin{cases}\mathrm{k}^{\mathrm{A}}\left(\mathrm{r}_{i j, n}-R_0^{\mathrm{A}}\right) \hat{\mathbf{r}}_{i j, n} & \text { if } r_{i j, n}< R_0^{A} ; \\ 0 & \text { if } r_{i j, n} \geq R_0^{\mathrm{A}},\end{cases}
```


#### Repulsion
When two cell membranes come into contact, the repulsive force keeps them from merging with one another. $F_R$, the repulsive force between nodes in different cells is defined as
```math
\begin{gathered}\mathbf{F}_{i, m}^{\mathrm{R}}=\sum_n \sum_j \begin{cases}-\mathrm{k}^{\mathrm{R}}\left(r_{i j, n}-R_0^{\mathrm{R}}\right) \hat{\mathbf{r}}_{i j, n} & \text { if } r_{i j, n}< R_0^{\mathrm{R}} ; \\ 0 & \text { if } r_{i j, n} \geq R_0^{\mathrm{R}},\end{cases} \end{gathered}
```



#### Intercellular friction
The viscous drag due to intermembrane friction is proportional to the relative velocity tangential to the cell surface.

```math
F^{F,e}_{ij}= -\gamma_{ext} v_{ij}^{\tau}
```
 <br> <br> 

<div align="center">

| Parameter | Symbol | Type | Description |
|---|---|---|---|
| `repulsion_range` | $R_0^R$ | float | Cutoff repulsion distance (between nodes on different cells) |
| `attraction_range` | $R_0^A$ | float | Cutoff adhesion distance (between nodes on different cells) | 
| `repulsion_strength` | $k^R$ | float | Repulsion strength (repulsion stiffness = $k^R \times E$) |
| `attraction_strength` | $k^A$ | float | Adhesion strength (adhesion stiffness = $k^A \times E$) |
| `intercellular_damping` | $\gamma_{ext}$ | float | Damping coefficient due to intercellular friction |

</div> 

## 3. apoptosis

In the apoptosis process, cells loose their internal pressure and shrink. This shrinkage is modeled by changing the growth rate (positive) to a shrink rate (negative). After the volume of these cells pass a certain threshold, they are removed from the
system.


<div align="center">

| Parameter | Type | Description |
|-----------|------|-------------|
| `apoptosis` | bool | Determines whether to include apoptosis in the simulation |
| `popToStartApo` | int | Number of cells we should have in the system to start apoptosis |
| `Apo_ratio` | float | Apoptosis probability|
| `squeeze_rate` | float | Rate of shrinkage (replaces growth rate) |
| `apoptosis_Vol` | float | Threshold volume below which cells are removed |
| `Create_wound` | bool | Determines whether to create a wound in the tissue (default: Spherical) |
| `Ephitelial_wound_layered` | bool | Creates a cylindrical wound in epithelial tissue |
| `epi_wound_Zratio` | float | height of the epithelial wound|
| `wound_radius` | float | Radius of the wound to be created ($R_{wound}$) |
| `Wound_creation_time` | int | Time step at which the wound is created |
| `Wound_Induced_Division_V` | float | Cell division volume after wounding|
| `WI_growth_rate` | float | Growth rate after wounding |
| `WI_viscotic_damping` | float | Intercellular damping after wounding |
| `WI_gamma_visc` | float | Medium friction after wounding |

</div>


## 4. Extracellular Matrix (ECM)

The ECM is modeled as a network of nodes connected by springs. The nodes in the ECM interact with each other and with the cells through adhesion, repulsion, and friction forces. The parameters governing the properties of the ECM include:

<div align="center">

| Parameter | Type | Description |
|-----------|------|-------------|
| `Add-ECM` | bool | Determines whether to include Extracellular Matrix (ECM) in the simulation |
| `Max_ECM_nodes` | int | Maximum number of nodes in the ECM, for constructing an array |
| `Div_size` | float | Division size in ECM |
| `Buffer_size` | int | Buffer size for ECM |
| `MaxNeighList` | int | Maximum number of cells we can have in a neighbor list for ECM interactions |
| `mass` | float | The mass of each node in the ECM |
| `stiffness_min` | float | Minimum stiffness of bonds in ECM (stiffness varies across ECM) |
| `stiffness_max` | float | Maximum stiffness of bonds in ECM |
| `angleConstant` | float | Bond stiffness constant for ECM |
| `vis_damp` | float | Inter-node friction in ECM |
| `gamma_env` | float | Environment friction for ECM |
| `vis_ecm_cell` | float | Friction between ECM and cell |
| `attraction_range_ecm` | float | Attraction range between two neighboring nodes in ECM |
| `repulsion_range_ecm` | float | Repulsion range between two neighboring nodes in ECM |
| `attraction_strength_ecm` | float | Attraction strength for ECM interactions (Attraction stiffness = $E \times k^A_{ECM}$) |
| `repulsion_strength_ecm` | float | Repulsion strength for ECM interactions (Repulsion stiffness = $E \times k^R_{ECM}$) |
| `write_traj_Ecm_file` | bool | Determines whether to write ECM trajectory file |

</div>


## 5. Division 
We assume that cells divide symmetrically through their centers of mass and asymmetrically by placing the division plane off-center.


<div align="center">

| Parameter | Type | Description |
|-----------|------|-------------|
| `Random_Div_Rule` | bool | Determines whether to randomly choose the division plane |
| `along_Major_axis` | bool | Determines whether to divide along the major axis of the cell |
| `Fibre` | bool | Determines whether to use the fiber-based division method |
| `useDivPlaneBasis` | bool | Determines whether to fix one vector in the division plane |
| `divPlaneBasisX` | float | X value of the fixed vector in the division plane |
| `divPlaneBasisY` | float | Y value of the fixed vector in the division plane |
| `divPlaneBasisZ` | float | Z value of the fixed vector in the division plane |
| `Rotation_angle` | float | Rotation angle for the division plane  |
| `Rotation_rate` | float | Rate at which the division plane rotates during the division process |
| `asymDivision` | bool | Determines whether to perform asymmetric division |

</div>


## 6. New cell characteristics


<div align="center">

| Parameter | Type | Description |
|-----------|------|-------------|
| `useDifferentCell` | bool | Determines whether to include a different type of cell with different properties in the simulation |
| `SizeFactor` | float | Size ratio of the new type of cell compared to the original cell |
| `StiffFactor2` | float | Stiffness factor for the new cell type|
| `GrowthRate` | float | Growth rate for the new cell type |
| `divisionV` | float | Division volume threshold for the new cell type |
| `medium_viscosity2` | float | Medium viscosity for the new cell type |
| `intercellular_damping2` | float | Intercellular damping for the new cell type |
| `Apo_rate2` | float | Apoptosis rate for the new cell type |
| `squeeze_rate2` | float | Shrinkage rate for the new cell type during apoptosis |
| `numberOfCells` | int | Number of initial cells that will be of the new type |
| `fractionOfCells` | float | Fraction of cells that will be of the new type |
| `closenessToCenter` | float | Radius from the center of the box within which cells will be turned into the new type |
| `Yseparation` | float | Y position threshold below which cells will be turned into the new type |
| `chooseRandomCellindices` | bool | Determines whether to randomly select cells to be of the new type |
| `daughtSame` | bool | Determines whether the daughter cell will have the same properties as te motter |
| `duringGrowth` | bool | Determines whether to apply the new cell properties during the growth phase |
| `recalc_r0` | bool | Determines whether to recalculate the equilibrium bond length for the new cell type |

</div>


## 7. Boundary conditions and simulation box specifications

### Boundary conditions

<div align="center">

| Boundary Condition | Type | Description |
|--------------------|------|-------------|
| `Rigid walls` | bool | Rigid simulation box|
| `Periodic Boundary Conditions (PBCs)` | bool | Use periodic boundary conditions |
| `Lees Edwards Boundary Conditions (LEbc)` | bool | Use Lees Edwards boundary conditions |
| `Rigid Box in Z direction` | bool | Have rigid walls in the Z direction |
| `Rigid Box in Y direction` | bool | Have rigid walls in the Y direction |
| `Rigid Box in X direction` | bool | Have rigid walls in the X direction |

</div>

### Box specifications 

<div align="center">

| Parameter | Type | Description |
|-----------|------|-------------|
| `threshDist` | float | Threshold distance for cell-wall interactions |
| `box_len_x` | float | Length of the simulation box in the x direction |
| `box_len_y` | float | Length of the simulation box in the y direction |
| `box_len_z` | float | Length of the simulation box in the z direction |
| `BoxMin_x` | float | Minimum x coordinate of the simulation box (usually set to 0.0) |
| `BoxMin_y` | float | Minimum y coordinate of the simulation box (usually set to 0.0) |
| `BoxMin_z` | float | Minimum z coordinate of the simulation box (usually set to 0.0) |
| `dom_len` | float | Cell diameter, used for neighbor list construction |

</div>

### Cell (particle) placement style

<div align="center">

| Placement Style | Type | Description |
|-----------------|------|-------------|
| `Flat box` | bool | Position the z position of all cells in the middle of the box |
| `Line` | bool | Position cells in a line |
| `Line Center` | bool | If "line" is set to one, It sets the x position of the COM of cells to the center of the box |
| `Plane` | bool | Position all cells on a plane with fixed Z |
| `Random in 3D` | bool | Give the initial cells random positions in 3D |
| `Random in 2D` | bool | Give the initial cells random positions in 2D (fixed Z) |
| `line`| bool | Position cells in a line |
| `plane` | bool | Position all cells on a plane with fixed Z |

</div>

### Cell-wall adhesive interactions and surface friction

<div align="center">

| Parameter | Type | Description |
|-----------|------|-------------|
| `wall_adhesion` | bool | Determines whether to include cell-wall adhesive interactions |
| `LJ9_3_epsilon` | float | Depth of the potential well for cell-wall interactions (Lennard-Jones 9-3 potential) |
| `LJ9_3_sigma` | float | Finite distance at which the inter-particle potential is zero|
| `Have_Surface_friction` | bool | Determines whether to include surface friction between cells and walls |
| `surface_friction_coeff` | float | Coefficient of friction for surface interactions |

</div>

### Compressing the system

<div align="center">

| Parameter | Type | Description |
|-----------|------|-------------|
| `compress` | bool | Determines whether to compress the system (default: with two walls)|
| `compression_ratio` | float | Ratio by which to compress the system (e.g., 0.5 for 50% compression) |
| `Compression_step_size` | float | Boundary movement per time step during compression |
| `Single_wall` | bool | Determines whether to apply compression from one side |

</div>

### Pinning and impurities

<div align="center">

| Parameter | Type | Description |
|-----------|------|-------------|
| `impurity` | bool | Determines whether to include pinned impurities in the simulation |
| `impurityNum` | int | Number of pinned impurities to include in the simulation |

</div>


## 8. Random number Generator

<div align="center">

| Parameter | Type | Description |
|-----------|------|-------------|
|add_rands| bool | Include random forces in the simulation |
|Levy_flight| bool | Use Levy flight distribution for random forces |
|Levy_alpha| float | Alpha parameter for Levy flight distribution |
|Gaussian| bool | Use Gaussian distribution for random forces |
|rand_seed| int | Seed for random number generator (use time if < 0) |
|rand_scale_factor| float | Scale factor for random forces |

</div>


## 9. Colloidal Dynamics

<div align="center">

| Parameter | Type | Description |
|-----------|------|-------------|
| `colloidal_dynamics` | bool | Determines whether to simulate colloidal dynamics in the system |
| `dispersity` | bool | Determines whether to include size dispersity in the colloidal particles |
| `dispersity_max` | float | Maximum size of particles in the system |
| `dispersity_min` | float | Minimum size of particles in the system |
| `rand_vel` | bool | Determines whether to assign random initial velocities to the colloidal particles |
| `Two_Components` | bool | Determines whether to include two different types of colloidal particles in the system |
| `SizeFactor` | float | Size ratio of the second type of colloidal particle compared to the first type |
| `Friction` | float | Friction coefficient for the colloidal particles |
| `Fraction` | float | Fraction of colloidal particles that are of the second type |
| `Xratio` | float | Scaling factor for the x dimension of non-spherical colloidal particles |
| `Yratio` | float | Scaling factor for the y dimension of non-spherical colloidal particles |
| `Zratio` | float | Scaling factor for the z dimension of non-spherical colloidal particles |
| `RandInitDir` | bool | Determines whether to assign random initial directions to the colloidal particles |
| `ReadInitialConf` | bool | Determines whether to read the initial configuration of colloidal particles from a file |
| `Compressor` | bool | Determines whether to use a compressor to achieve a specific density of colloidal particles |
| `Compress_Value_X` | float | Scaling factor for compressing the system in the x direction |
| `Compress_Value_Y` | float | Scaling factor for compressing the system in the y direction |
| `Compress_Value_Z` | float | Scaling factor for compressing the system in the z direction |

</div>

## 10. Fluid forces

<div align="center">

| Parameter | Type | Description |
|-----------|------|-------------|
| `LateralForce` | bool | Determines whether to apply a lateral force to the cells |
| `Constant_pressure` | float | Magnitude of the constant pressure applied to the cells |

</div>

## 11. Counting Cells (legacy):

<div align="center">

| Parameter | Type | Description |
|-----------|------|-------------|
| `countcells` | bool | Count the number of cells in the system during the simulation |
| `mit-index_file_name` | string | Name of the mitosis file |
| `count_only_internal_cells?` | bool | Determines whether to count only internal cells within a certain radius from the center of the box |
| `radius_cutoff` | float | Radius cutoff for counting internal cells |
| `overwrite_mit_ind_file?` | bool | Overwrite the mitosis index file |
| `cell_count_int` | int | Interval at which to count cells during the simulation |

</div>

## 12. Population control (legacy):

<div align="center">

| Parameter | Type | Description |
|-----------|------|-------------|
| `doPopModel` | bool | Determines whether to include a population model in the simulation
| `totalFood` | float | Total amount of food available in the system for the population model |
| `regular_consumption` | float | Amount of food consumed by each cell at each time step|
| `division_consumption` | float | Additional amount of food consumed by a cell during division |
| `death_release_food` | float | Amount of food released back into the system when a cell dies |
| `haylimit` | float | Threshold of food availability below which cells will start dying |
| `cellLifeTime` | int | Maximum lifetime of a cell in the population model |

</div>


## 13. System shape

"System_shape": {
        "_comment": "Initializing a rigid boundary in the shape of a sphere",
        "Sphere": false,
        "_comment2": "Cutting out a sphere from the initial configuration - best used after resart",
        "Cut_out_Sphere": false,
        "Sphere_radius": 12.86
    }
}

<div align="center">

| Parameter | Type | Description |
|-----------|------|-------------|
| `Sphere` | bool | Initialize a rigid boundary in the shape of a sphere (repulsion $\propto$ `Constant_pressure`) |
| `Cut_out_Sphere` | bool | Cut out a sphere from the initial configuration - best used after restart (removes cells within `Sphere_radius`) |
| `Sphere_radius` | float | Radius of the sphere for the above two parameters |

</div>

<br>  <br> 


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


More details on thie software can be found in [this link](https://ir.lib.uwo.ca/etd/9239/).

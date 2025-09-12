# Changelog  

All notable changes to this project will be documented here.  

## September 2014 - September 2023  *(Pranav Madhikar and Mahmood Mazarei)*
### Implementations
+ Apoptosis *(by Mahmood Mazarei)*
+ ECM *(by Mahmood Mazarei)*
+ Different division styles (Symmetric – Asymmetric)  
+ Different division planes (Random, Selected plane, Rotation)  
+ Second cell type  
+ Colloidal dynamics *(by Mahmood Mazarei)*
+ Multiple boundary conditions (Rigid walls, PBC, Lees–Edwards)  
+ Multiple initialization positioning  
+ Gaussian randomness  
+ Multi GPU implementation *(by Mahmood Mazarei)*
  

---

## September 2023 - August 2024 *(by Yasamin Modabber)* 
### Added  
- Added a documentation for the input file

### Changed  
- Changed Angle Constant from a fixed value to an input in the inp.json file
- Updated the Readme file

### Fixed  
-  Fixed arithmetic error: division by zero in angle force calculations


## September 2024 - August 2025 *(by Yasamin Modabber)* 
### Added  
- Wound creation  
- New boundary condition: Adhesive walls (Lennard–Jones 9:3 potential)  
- Lévy-flight type random walks 
- Fluid forces (lateral pressure, hydrostatic pressure)

### Changed  
- Nodal growth force: $\mathbf{F}^P = 3P\hat{\mathbf{n}} \, \rightarrow 1000AP\hat{\mathbf{n}}$
- Updated makefile 
- Turned off correct_VCOM

### Fixed  
- Fixed celldiv.py rendering for single cell type case


## September 2025 *(by Yasamin Modabber)* 
### Added  
- [Placeholder: TBA ] 

### Changed  
- Removed "doAdaptive_dt" 

### Fixed  
- Fixed unphysical net force due to internal pressure
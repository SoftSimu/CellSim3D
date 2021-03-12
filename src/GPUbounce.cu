//#define FORCE_DEBUG
//#define PRINT_VOLUMES
//#define TURNOFF_RAN
//#define DEBUG_RAND
//#define OUTPUT_ADP_ERROR
//#degine RO_DEBUG
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <locale.h>
#include <algorithm>
#include <iostream>
#include <fstream>
#include <streambuf>
#include <cstring>
#include <string>

#include <cuda.h>
#include <curand.h>
#include <vector_functions.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/fill.h>
#include <thrust/sort.h>
#include <thrust/extrema.h>
//#include "helper_cuda.h"
#include "postscript.h"
#include "marsaglia.h"
//#include "IntegrationKernels.h"
#include "RandomVector.h"
#include "VectorFunctions.hpp"
#include "AdaptiveTimeKernels.cuh"

#include "json/json.h"


void CudaFailure();

#define CudaErrorCheck() { \
        cudaError_t e = cudaPeekAtLastError();        \
        if (e!=cudaSuccess){                                            \
            printf("Cuda failure in %s, line %d, code %d %s\n", __FILE__, __LINE__, e, cudaGetErrorString(e)); \
            exit(0); \
        }\
    }



// the three nearest neighbours of C180 atoms
int   C180_nn[3*192];
int   C180_sign[180];
// device: the three nearest neighbours of C180 atoms
int   *d_C180_nn;
int   *d_C180_sign;
int   CCI[2][271];       // list of nearest neighbor carbon pairs in the fullerne
// number of pairs = 270
int   C180_56[92*7];     // 12 lists of atoms forming pentagons 1 2 3 4 5 1 1 and
// 80 lists of atoms forming hexagons  1 2 3 4 5 6 1
int   *d_C180_56;


float mass;                                           //  M
float repulsion_range,    attraction_range;        //  LL1, LL2
float repulsion_strength, attraction_strength;     //  ST1, ST2
float viscotic_damping, internal_damping;          //  C, DMP    
float divVol;
float ApoVol;
float gamma_visc;
float shear_rate;
float Pshift;
float Vshift;
float maxPressure;
float minPressure;
float rMax;
float squeeze_rate;
int Restart;
int Laststep = 0;
int Lastframe = 0;
int   Time_steps;
int   trajWriteInt; // trajectory write interval

float delta_t;

//__constant__ float d_dt;

float dt_max;
float dt_tol;
bool doAdaptive_dt;
float c1 = 0; float c2 = 0; 

bool write_cont_force=false;
bool write_vel_file = false;
char forces_file[256];
int   overWriteMitInd; // 0 No, 1 yes
const char* ptrajFileName;
char trajFileName[256];
bool binaryOutput; 
char mitIndFileName[256]; 


bool asymDivision;
float* asym;
float* d_asym;
bool checkSphericity; 
bool useDivPlaneBasis;
float divPlaneBasis[3]; 



int   countOnlyInternal; // 0 - Count all new cells
                         // 1 - Count only the cells born within 0.6Rmax from
                         //     the center of mass of the system
float radFrac; 	// The factor to count cells within a raduys (<Rmax)
int newCellCountInt; // Interval at which to count the divided cells
int equiStepCount;
bool countCells;

// equilibrium length of springs between fullerene atoms
float* d_R0;
float* h_R0;

bool constrainAngles;
angles3* theta0;


float L1  = 3.0f;       // the initial fullerenes are placed in
			// an X x Y grid of size L1 x L1


float *d_volume;
float *volume;
float *d_area; 
float *area; 


char* cell_div;
char* d_cell_div;
int num_cell_div;
int* cell_div_inds;

char* cell_Apo;
char* d_cell_Apo;
int num_cell_Apo;
int* cell_Apo_inds;



// Params related to population modelling
int doPopModel;
char* didCellDie;
float totalFood;
float* d_totalFood;
int haylimit;
int cellLifeTime;
float cellFoodCons; // baseline food consumption
float cellFoodConsDiv; // Extra good consumption when cell divides
float cellFoodRel; // Food released when cell dies (should < total consumed food)
float maxPop;
 

float3 boxMax;
float3 BoxMin;
float3 BoxCen;
bool flatbox; 
bool LineCenter; 
bool useRigidSimulationBox;
float threshDist;
bool usePBCs; 
bool useLEbc;
bool useRigidBoxZ; 
bool useRigidBoxY; 
bool rand_pos;
bool impurity;
int impurityNum;
bool line;
bool plane;
float L  = 2.5f;  

int No_of_threads; // ie number of staring cells
int Side_length;
int ex, ey;



// randomness parameters

bool add_rands;
int rand_seed;
int rand_dist;
float rand_scale_factor;
curandState *d_rngStates;
unsigned int *d_seeds; 



float  *X,  *Y,  *Z;     // host: atom positions
//float *d_XP, *d_YP, *d_ZP;     // device: time propagated atom positions
float  *d_X,  *d_Y,  *d_Z;     // device: present atom positions
float* d_velListX, *d_velListY, *d_velListZ; 
float* velListX, *velListY, *velListZ; 


R3Nptrs d_fConList;
R3Nptrs d_fDisList;
R3Nptrs d_fRanList; 
R3Nptrs d_ExtForces;
R3Nptrs h_contactForces;
R3Nptrs h_ExtForces;


float DL;
float3 DLp;
int Xdiv, Ydiv, Zdiv;

//int *d_NoofNNlist;
int *d_NNlist;
int *NoofNNlist;
int *NNlist;


bool correct_com = false;
bool correct_Vcom = false;
int reductionblocks;

float *d_CMx, *d_CMy, *d_CMz;
float *CMx, *CMy, *CMz;
float *d_VCMx, *d_VCMy, *d_VCMz;
float *VCMx, *VCMy, *VCMz;
float *d_SysCx, *d_SysCy, *d_SysCz; 

R3Nptrs h_sysCM;
R3Nptrs d_sysCM;
R3Nptrs d_sysVCM;
R3Nptrs h_sysVCM;


//float Pressure;          // pressure
//float Temperature;       // equation of state relates Pressure and Temperature

int  No_of_C180s;        // the global number of C180 fullerenes
int  No_of_C180s_in;     // the number of C180s near the center of mass of the system
int MaxNoofC180s;
int NewCellInd; 

float *ran2;             // host: ran2[]
float *d_ran2;           // device: ran2[], used in celldivision

int *NDIV;               // # of divisions

// Parameters related to division

long int GPUMemory;
long int CPUMemory;


int frameCount = 1;


int Orig_No_of_C180s;
 

bool apoptosis;
float Apo_rate;
int popToStartApo;
bool WithoutApo;
int NumApoCell;
int NumRemoveCell;



bool colloidal_dynamics;
bool dispersity;
bool rand_vel;
float Xratio;
float Yratio;
float Zratio;
float shapeLim;
bool RandInitDir;



bool useDifferentCell;
float* ScaleFactor;
float* d_ScaleFactor;
float* DivisionVolume;
float* d_DivisionVolume;
float* gamma_env;
float* d_gamma_env;
float* viscotic_damp;
float* d_viscotic_damp;
float* d_Growth_rate;
float* Growth_rate;
float* d_Youngs_mod;
float* youngsModArray;
float *d_pressList;
float *pressList;
int* d_resetIndices;
int* resetIndices; 
int* CellINdex;
int* d_CellINdex;


float SizeFactor;
float Stiffness2;
float stiffness1;
float Youngs_mod; 
float gRate;
float divisionV;
float gEnv;
float gVis;
float Apo_rate2;
float squeeze_rate2;
int   numberOfCells;
float fractionOfCells;
float closenessToCenter;
bool  chooseRandomCellIndices;
bool daughtSame;
bool duringGrowth;
bool recalc_r0; 


R3Nptrs DivPlane;
R3Nptrs d_DivPlane;
float f_range;

int main(int argc, char *argv[])
{


  int globalrank;
  int step = 0;
  int noofblocks, threadsperblock;
  int newcells;
  
  NumApoCell = 0;
  NumRemoveCell = 0;
  
  
  FILE *outfile;
  FILE *trajfile; // pointer to xyz file
  cudaError_t myError;

  int* dividingCells; //Cells that are about to divide
  int* totalCells; // No. of cells at every Dividing_steps


  int* num_new_cells_per_step;
  int countOffset = 0;

  //int min_no_of_cells = 10;

  printf("CellDiv version 0.9\n");

  if ( argc !=4 )
  {
      printf("Usage: CellDiv no_of_threads inpFile.json gpuid\n");
      return(0);
  }
  

  No_of_threads = atoi(argv[1]);

  char inpFile[256];
  strcpy(inpFile, argv[2]);

  if ( read_json_params(inpFile)          != 0 ) return(-1);

  printf("%d\n", MaxNoofC180s); 
  
  Vshift = shear_rate*boxMax.x;
  Pshift = 0;
  
  float Vscale;
  float scale;
	
  Vscale = 1.0/(Xratio*Yratio*Zratio);
  scale = pow(Vscale, 1.0/3.0);
	
  Xratio *= scale;
  Yratio *= scale;
  Zratio *= scale;

  
  if((Xratio >= Yratio) && (Xratio >= Zratio)){
        shapeLim = Xratio;
  }else if ((Yratio >= Xratio) && (Yratio >= Zratio)){
        shapeLim = Yratio;
  }else{
        shapeLim = Zratio;
  }

  if(!colloidal_dynamics) shapeLim = 1.0f;
  
  f_range = (attraction_range + 0.9*shapeLim) * (attraction_range + 0.9*shapeLim);
  	
  if ( line ) {
  
  	int LineCell = 0;
  	LineCell = (int) ((boxMax.x - BoxMin.x)/L);
  	printf(" Max Number of initial cells in line is:   %d\n", LineCell); 
  	if ( No_of_threads > LineCell ) {
  	printf(" Number of initial cells in line is greater than Max \n ");
  	return(-1);
  	}	
  }	

  Side_length   = (int)( sqrt( (double)No_of_threads )+0.5);
  if ( No_of_threads > MaxNoofC180s // Side_length*Side_length != No_of_threads
      )
  {
      printf("Usage: Celldiv no_of_threads\n");
      printf("       no_of_threads should be a square, n^2, < %d\n", MaxNoofC180s);
      return(0);
  }

  if (!impurity) impurityNum = 0;	
  No_of_C180s      = No_of_threads + impurityNum;
  Orig_No_of_C180s = No_of_C180s;
  GPUMemory = 0L;
  CPUMemory = 0L;



  X = (float *)calloc(192*MaxNoofC180s,sizeof(float));
  Y = (float *)calloc(192*MaxNoofC180s,sizeof(float));
  Z = (float *)calloc(192*MaxNoofC180s,sizeof(float));
  velListX = (float *)calloc(192*MaxNoofC180s, sizeof(float)); 
  velListY = (float *)calloc(192*MaxNoofC180s, sizeof(float)); 
  velListZ = (float *)calloc(192*MaxNoofC180s, sizeof(float));
  youngsModArray = (float *)calloc(MaxNoofC180s, sizeof(float));
  Growth_rate = (float *)calloc(MaxNoofC180s, sizeof(float));
  pressList = (float *)calloc(MaxNoofC180s, sizeof(float));  
  CellINdex = (int *)calloc(MaxNoofC180s, sizeof(int));
  ScaleFactor = (float *)calloc(MaxNoofC180s, sizeof(float)); 
  DivisionVolume = (float *)calloc(MaxNoofC180s, sizeof(float));
  gamma_env = (float *)calloc(MaxNoofC180s, sizeof(float));
  viscotic_damp = (float *)calloc(MaxNoofC180s, sizeof(float));
  h_contactForces.x = (float *)calloc(192*MaxNoofC180s, sizeof(float));
  h_contactForces.y = (float *)calloc(192*MaxNoofC180s, sizeof(float));
  h_contactForces.z = (float *)calloc(192*MaxNoofC180s, sizeof(float));
  DivPlane.x = (float *)calloc(MaxNoofC180s, sizeof(float));
  DivPlane.y = (float *)calloc(MaxNoofC180s, sizeof(float));
  DivPlane.z = (float *)calloc(MaxNoofC180s, sizeof(float));
  h_ExtForces.x = (float *)calloc(192*MaxNoofC180s, sizeof(float));
  h_ExtForces.y = (float *)calloc(192*MaxNoofC180s, sizeof(float));
  h_ExtForces.z = (float *)calloc(192*MaxNoofC180s, sizeof(float));
  CMx = (float *)calloc(MaxNoofC180s, sizeof(float));
  CMy = (float *)calloc(MaxNoofC180s, sizeof(float));
  CMz = (float *)calloc(MaxNoofC180s, sizeof(float));
  VCMx = (float *)calloc(MaxNoofC180s, sizeof(float));
  VCMy = (float *)calloc(MaxNoofC180s, sizeof(float));
  VCMz = (float *)calloc(MaxNoofC180s, sizeof(float));
  area= (float *)calloc(MaxNoofC180s, sizeof(float));
  cell_div = (char *)calloc(MaxNoofC180s, sizeof(char));
  cell_div_inds = (int *)calloc(MaxNoofC180s, sizeof(int));
  cell_Apo = (char *)calloc(MaxNoofC180s, sizeof(char));
  cell_Apo_inds = (int *)calloc(MaxNoofC180s, sizeof(int));
  NoofNNlist = (int *)calloc( 1024*1024,sizeof(int));
  NNlist =  (int *)calloc(32*1024*1024, sizeof(int));
  asym = (float *)calloc(MaxNoofC180s, sizeof(float));
  resetIndices = (int *)calloc(MaxNoofC180s, sizeof(int));
  h_sysCM.x = (float *)calloc(1, sizeof(float));
  h_sysCM.y = (float *)calloc(1, sizeof(float));
  h_sysCM.z = (float *)calloc(1, sizeof(float));
  h_sysVCM.x = (float *)calloc(1, sizeof(float));
  h_sysVCM.y = (float *)calloc(1, sizeof(float));
  h_sysVCM.z = (float *)calloc(1, sizeof(float));
//volume= (float *)calloc(MaxNoofC180s, sizeof(float));  
//  SysCx = (float *) calloc(1024 , sizeof(float));
//  SysCy = (float *) calloc(1024 , sizeof(float));
//  SysCz = (float *) calloc(1024 , sizeof(float));



  CPUMemory += 6L*192L*MaxNoofC180s*sizeof(float);
  CPUMemory += MaxNoofC180s*10L*sizeof(float);
  CPUMemory += MaxNoofC180s*7L*sizeof(float);
  CPUMemory += MaxNoofC180s*sizeof(float);
  CPUMemory += 3L*MaxNoofC180s*sizeof(float);
  CPUMemory += 6L*1024L*sizeof(float);
  CPUMemory += MaxNoofC180s*sizeof(char);
  CPUMemory += MaxNoofC180s*sizeof(int);
  CPUMemory += MaxNoofC180s*sizeof(int); 
  CPUMemory += 3*180*sizeof(float);
  CPUMemory += 2*MaxNoofC180s*sizeof(int); 



  cudaDeviceProp deviceProp = getDevice();
  if (cudaSuccess != cudaSetDevice(atoi(argv[3]))){
      CudaErrorCheck();
      printf("Could not set to divice %d\n", atoi(argv[3]));
      return -1;
  }


  thrust::device_vector<float> d_volumeV(MaxNoofC180s);
  thrust::host_vector<float> h_volume(MaxNoofC180s);
  thrust::fill(d_volumeV.begin(), d_volumeV.end(), 0.f);
  d_volume = thrust::raw_pointer_cast(&d_volumeV[0]);
  volume = thrust::raw_pointer_cast(&h_volume[0]);
  
  thrust::host_vector<angles3> h_theta0(192);
  thrust::device_vector<angles3> d_theta0V(192);
  angles3* d_theta0 = thrust::raw_pointer_cast(&d_theta0V[0]);
  theta0 = thrust::raw_pointer_cast(&h_theta0[0]);


  h_R0 = (float *)calloc(192*3, sizeof(float));
  for (int i =  0; i < MaxNoofC180s; ++i) ScaleFactor[i] = 1.0;

  //if ( read_global_params()               != 0 ) return(-1);
  if (Restart == 1 ) if( ReadRestartFile() != 0 ) return(-1); 
  if (generate_random(Orig_No_of_C180s)  != 0 ) return(-1);
  if (DispersityFunc(Orig_No_of_C180s) != 0 ) return(-1);
  if (Restart == 0 ) if ( initialize_C180s(Orig_No_of_C180s) != 0 ) return(-1);
  if ( read_fullerene_nn()                != 0 ) return(-1);
  
 
  // empty the psfil from previous results
  outfile = fopen("psfil","w");
  if ( outfile == NULL ) {printf("Unable to open file psfil\n");return(-1);}
  fclose(outfile);

  /* PM
     Allocate memory for the dividingCells array that will be used to
     calculate the mitotic index.
  */

  dividingCells = (int *)calloc((Time_steps/newCellCountInt), sizeof(int));
  totalCells = (int *)calloc((Time_steps/newCellCountInt), sizeof(int));
  num_new_cells_per_step = (int *)calloc(Time_steps, sizeof(int));
  


  CPUMemory += (2L*(long)(Time_steps/newCellCountInt) + 1L + (long)Time_steps) * sizeof(int);
  CPUMemory += (long)MaxNoofC180s * sizeof(char);





  if ( cudaSuccess != cudaMalloc((void **)&d_sysVCM.x, sizeof(float))) return -1;
  if ( cudaSuccess != cudaMalloc((void **)&d_sysVCM.y, sizeof(float))) return -1;
  if ( cudaSuccess != cudaMalloc((void **)&d_sysVCM.z, sizeof(float))) return -1;  
  if ( cudaSuccess != cudaMalloc((void **)&d_sysCM.x, sizeof(float))) return -1;
  if ( cudaSuccess != cudaMalloc((void **)&d_sysCM.y, sizeof(float))) return -1;
  if ( cudaSuccess != cudaMalloc((void **)&d_sysCM.z, sizeof(float))) return -1;    
  if ( cudaSuccess != cudaMalloc( (void **)&d_C180_nn, 3*192*sizeof(int))) return(-1);
  if ( cudaSuccess != cudaMalloc( (void **)&d_C180_sign, 180*sizeof(int))) return(-1);
  if ( cudaSuccess != cudaMalloc( (void **)&d_X  , 192*MaxNoofC180s*sizeof(float))) return(-1);
  if ( cudaSuccess != cudaMalloc( (void **)&d_Y  , 192*MaxNoofC180s*sizeof(float))) return(-1);
  if ( cudaSuccess != cudaMalloc( (void **)&d_Z  , 192*MaxNoofC180s*sizeof(float))) return(-1);
  if ( cudaSuccess != cudaMalloc( (void **)&d_CMx , MaxNoofC180s*sizeof(float))) return(-1);
  if ( cudaSuccess != cudaMalloc( (void **)&d_CMy , MaxNoofC180s*sizeof(float))) return(-1);
  if ( cudaSuccess != cudaMalloc( (void **)&d_CMz , MaxNoofC180s*sizeof(float))) return(-1);
  if ( cudaSuccess != cudaMalloc( (void **)&d_VCMx ,          MaxNoofC180s*sizeof(float))) return(-1);
  if ( cudaSuccess != cudaMalloc( (void **)&d_VCMy ,          MaxNoofC180s*sizeof(float))) return(-1);
  if ( cudaSuccess != cudaMalloc( (void **)&d_VCMz ,          MaxNoofC180s*sizeof(float))) return(-1);
  if ( cudaSuccess != cudaMalloc( (void **)&d_area ,       MaxNoofC180s*sizeof(float))) return(-1);
  if ( cudaSuccess != cudaMalloc( (void **)&d_cell_div ,     MaxNoofC180s*sizeof(char))) return(-1);
  if ( cudaSuccess != cudaMalloc( (void **)&d_cell_Apo ,     MaxNoofC180s*sizeof(char))) return(-1); 
  if ( cudaSuccess != cudaMalloc( (void **)&d_C180_56,       92*7*sizeof(int))) return(-1);
  if ( cudaSuccess != cudaMalloc( (void **)&d_ran2 , 10000*sizeof(float))) return(-1);
  if ( cudaSuccess != cudaMalloc( (void **)&d_pressList, MaxNoofC180s*sizeof(float))) return(-1);
  if ( cudaSuccess != cudaMalloc( (void **)&d_resetIndices, MaxNoofC180s*sizeof(int))) return(-1);
  if ( cudaSuccess != cudaMalloc( (void **)&d_Youngs_mod, MaxNoofC180s*sizeof(float))) return(-1);
  if ( cudaSuccess != cudaMalloc( (void **)&d_Growth_rate, MaxNoofC180s*sizeof(float))) return(-1);
  if ( cudaSuccess != cudaMalloc( (void **)&d_CellINdex, MaxNoofC180s*sizeof(int))) return(-1);
  if ( cudaSuccess != cudaMalloc( (void **)&d_ScaleFactor, MaxNoofC180s*sizeof(float))) return(-1); 
  if ( cudaSuccess != cudaMalloc( (void **)&d_DivisionVolume, MaxNoofC180s*sizeof(float))) return(-1); 
  if ( cudaSuccess != cudaMalloc( (void **)&d_gamma_env, MaxNoofC180s*sizeof(float))) return(-1); 
  if ( cudaSuccess != cudaMalloc( (void **)&d_viscotic_damp, MaxNoofC180s*sizeof(float))) return(-1); 
  if ( cudaSuccess != cudaMalloc( (void **)&d_R0, 192*3*sizeof(float))) return(-1);
  if ( cudaSuccess != cudaMalloc((void **)&d_velListX, 192*MaxNoofC180s*sizeof(float))) return -1; 
  if ( cudaSuccess != cudaMalloc((void **)&d_velListY, 192*MaxNoofC180s*sizeof(float))) return -1; 
  if ( cudaSuccess != cudaMalloc((void **)&d_velListZ, 192*MaxNoofC180s*sizeof(float))) return -1;
  if ( cudaSuccess != cudaMalloc((void **)&d_ExtForces.x, 192*MaxNoofC180s*sizeof(float))) return -1;
  if ( cudaSuccess != cudaMalloc((void **)&d_ExtForces.y, 192*MaxNoofC180s*sizeof(float))) return -1;
  if ( cudaSuccess != cudaMalloc((void **)&d_ExtForces.z, 192*MaxNoofC180s*sizeof(float))) return -1;
  if ( cudaSuccess != cudaMalloc((void **)&d_fConList.x, 192*MaxNoofC180s*sizeof(float))) return -1;
  if ( cudaSuccess != cudaMalloc((void **)&d_fConList.y, 192*MaxNoofC180s*sizeof(float))) return -1;
  if ( cudaSuccess != cudaMalloc((void **)&d_fConList.z, 192*MaxNoofC180s*sizeof(float))) return -1;
  if ( cudaSuccess != cudaMalloc((void **)&d_fDisList.x, 192*MaxNoofC180s*sizeof(float))) return -1;
  if ( cudaSuccess != cudaMalloc((void **)&d_fDisList.y, 192*MaxNoofC180s*sizeof(float))) return -1;
  if ( cudaSuccess != cudaMalloc((void **)&d_fDisList.z, 192*MaxNoofC180s*sizeof(float))) return -1;
  if ( cudaSuccess != cudaMalloc((void **)&d_fRanList.x, 192*MaxNoofC180s*sizeof(float))) return -1;
  if ( cudaSuccess != cudaMalloc((void **)&d_fRanList.y, 192*MaxNoofC180s*sizeof(float))) return -1;
  if ( cudaSuccess != cudaMalloc((void **)&d_fRanList.z, 192*MaxNoofC180s*sizeof(float))) return -1;
  if ( cudaSuccess != cudaMalloc((void **)&d_DivPlane.x, MaxNoofC180s*sizeof(float))) return -1;
  if ( cudaSuccess != cudaMalloc((void **)&d_DivPlane.y, MaxNoofC180s*sizeof(float))) return -1;
  if ( cudaSuccess != cudaMalloc((void **)&d_DivPlane.z, MaxNoofC180s*sizeof(float))) return -1;
  if ( cudaSuccess != cudaMalloc((void **)&d_asym, MaxNoofC180s*sizeof(float))) return -1;
  if ( cudaSuccess != cudaMalloc((void **)&d_SysCx, 1024*sizeof(float))) return -1;
  if ( cudaSuccess != cudaMalloc((void **)&d_SysCy, 1024*sizeof(float))) return -1;
  if ( cudaSuccess != cudaMalloc((void **)&d_SysCz, 1024*sizeof(float))) return -1;



  cudaMemset(d_C180_nn, 0, 3*192*sizeof(int));
  cudaMemset(d_C180_sign, 0, 180*sizeof(int));
  CudaErrorCheck();

  cudaMemset(d_X, 0, 192*MaxNoofC180s*sizeof(float));
  cudaMemset(d_Y, 0, 192*MaxNoofC180s*sizeof(float));
  cudaMemset(d_Z, 0, 192*MaxNoofC180s*sizeof(float));
  CudaErrorCheck();

  cudaMemset(d_CMx, 0, MaxNoofC180s*sizeof(float));
  cudaMemset(d_CMy, 0, MaxNoofC180s*sizeof(float));
  cudaMemset(d_CMz, 0, MaxNoofC180s*sizeof(float));
  cudaMemset(d_VCMx, 0, MaxNoofC180s*sizeof(float));
  cudaMemset(d_VCMy, 0, MaxNoofC180s*sizeof(float));
  cudaMemset(d_VCMz, 0, MaxNoofC180s*sizeof(float));
  CudaErrorCheck();

  cudaMemset(d_R0, 0, 3*192*sizeof(float));
  cudaMemset(d_pressList, 0, MaxNoofC180s*sizeof(float));
  cudaMemset(d_Youngs_mod, 0, MaxNoofC180s*sizeof(float));
  cudaMemset(d_Growth_rate, 0, MaxNoofC180s*sizeof(float));
  cudaMemset(d_CellINdex, 0, MaxNoofC180s*sizeof(int));
  cudaMemset(d_ScaleFactor, 0, MaxNoofC180s*sizeof(int));
  cudaMemset(d_DivisionVolume, 0, MaxNoofC180s*sizeof(float));
  cudaMemset(d_gamma_env, 0, MaxNoofC180s*sizeof(float));
  cudaMemset(d_viscotic_damp, 0, MaxNoofC180s*sizeof(float));
  cudaMemset(d_area, 0, MaxNoofC180s*sizeof(float));
  CudaErrorCheck();

  cudaMemset(d_velListX, 0, 192*MaxNoofC180s*sizeof(float));
  cudaMemset(d_velListY, 0, 192*MaxNoofC180s*sizeof(float));
  cudaMemset(d_velListZ, 0, 192*MaxNoofC180s*sizeof(float));
  CudaErrorCheck();

  cudaMemset(d_fConList.x, 0, 192*MaxNoofC180s*sizeof(float));
  cudaMemset(d_fConList.y, 0, 192*MaxNoofC180s*sizeof(float));
  cudaMemset(d_fConList.z, 0, 192*MaxNoofC180s*sizeof(float));
  CudaErrorCheck();
  
  cudaMemset(d_fDisList.x, 0, 192*MaxNoofC180s*sizeof(float));
  cudaMemset(d_fDisList.y, 0, 192*MaxNoofC180s*sizeof(float));
  cudaMemset(d_fDisList.z, 0, 192*MaxNoofC180s*sizeof(float));
  CudaErrorCheck();
  cudaMemset(d_fRanList.x, 0, 192*MaxNoofC180s*sizeof(float));
  cudaMemset(d_fRanList.y, 0, 192*MaxNoofC180s*sizeof(float));
  cudaMemset(d_fRanList.z, 0, 192*MaxNoofC180s*sizeof(float));
  CudaErrorCheck();
  cudaMemset(d_ExtForces.x, 0, 192*MaxNoofC180s*sizeof(float));
  cudaMemset(d_ExtForces.y, 0, 192*MaxNoofC180s*sizeof(float));
  cudaMemset(d_ExtForces.z, 0, 192*MaxNoofC180s*sizeof(float));
  CudaErrorCheck();

  
  cudaMemset(d_DivPlane.x, 0, MaxNoofC180s*sizeof(float));
  cudaMemset(d_DivPlane.y, 0, MaxNoofC180s*sizeof(float));
  cudaMemset(d_DivPlane.z, 0, MaxNoofC180s*sizeof(float));
  CudaErrorCheck();
  
  
  cudaMemset(d_SysCx, 0, 1024*sizeof(float));
  cudaMemset(d_SysCy, 0, 1024*sizeof(float));
  cudaMemset(d_SysCz, 0, 1024*sizeof(float));
  CudaErrorCheck();
  
  
  //cudaMemcpyToSymbol(d_dt, &delta_t, sizeof(float),0, cudaMemcpyHostToDevice);


  if (cudaSuccess != cudaMemcpy(d_R0, h_R0, 3*192*sizeof(float), cudaMemcpyHostToDevice)) return -1; 

  //cudaMemcpy(d_pressList, pressList, MaxNoofC180s*sizeof(float), cudaMemcpyHostToDevice);
  


  if (colloidal_dynamics && rand_vel && !Restart ){
  	 
  	if ( initialize_Vel(Orig_No_of_C180s) != 0 ) return(-1);
  	cudaMemcpy(d_velListX, velListX, 192*No_of_C180s*sizeof(float), cudaMemcpyHostToDevice);
      	cudaMemcpy(d_velListY, velListY, 192*No_of_C180s*sizeof(float), cudaMemcpyHostToDevice);
  	cudaMemcpy(d_velListZ, velListZ, 192*No_of_C180s*sizeof(float), cudaMemcpyHostToDevice);	 
  	 
  }



  cudaMemcpy(d_C180_nn,   C180_nn,   3*192*sizeof(int),cudaMemcpyHostToDevice);
  cudaMemcpy(d_C180_sign, C180_sign, 180*sizeof(int),cudaMemcpyHostToDevice);
  cudaMemcpy(d_C180_56,   C180_56,   7*92*sizeof(int),cudaMemcpyHostToDevice);
  CudaErrorCheck();

  cudaMemcpy(d_cell_div, cell_div, MaxNoofC180s*sizeof(char), cudaMemcpyHostToDevice);
  CudaErrorCheck();
  cudaMemcpy(d_cell_Apo, cell_Apo, MaxNoofC180s*sizeof(char), cudaMemcpyHostToDevice);
  CudaErrorCheck();



if (Restart == 0) {	


  	for (int cell = 0; cell < MaxNoofC180s; cell++){
		if (colloidal_dynamics){
			pressList[cell] = 0;
		}else{
		 	pressList[cell] = minPressure; 
  		}
  	}
	
  	for (int i =  0; i < MaxNoofC180s; ++i){
       	if (colloidal_dynamics){
			Growth_rate[i] = 0;
		}else{
			Growth_rate[i] = rMax; 
  		}
  	}
	
	
  	for (int i =  0; i < MaxNoofC180s; ++i){
      		youngsModArray[i] = stiffness1; 
  	}
  
    	for (int i =  0; i < MaxNoofC180s; ++i){
      		DivisionVolume[i] = divVol; 
  	}
  	
  	for (int i =  0; i < MaxNoofC180s; ++i){
      		gamma_env[i] = gamma_visc; 
  	}
  	
  	for (int i =  0; i < MaxNoofC180s; ++i){
      		viscotic_damp[i] = viscotic_damping; 
  	}
  	
  
	if(!colloidal_dynamics && useDifferentCell) SecondCell(Orig_No_of_C180s);
  
 } // end of restart if else


  cudaMemcpy(d_X,  X, 192*No_of_C180s*sizeof(float),cudaMemcpyHostToDevice);
  cudaMemcpy(d_Y,  Y, 192*No_of_C180s*sizeof(float),cudaMemcpyHostToDevice);
  cudaMemcpy(d_Z,  Z, 192*No_of_C180s*sizeof(float),cudaMemcpyHostToDevice);
  CudaErrorCheck();
  cudaMemcpy(d_velListX, velListX, 192*No_of_C180s*sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_velListY, velListY, 192*No_of_C180s*sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_velListZ, velListZ, 192*No_of_C180s*sizeof(float), cudaMemcpyHostToDevice);
  CudaErrorCheck();
  cudaMemcpy(d_Youngs_mod, youngsModArray, MaxNoofC180s*sizeof(float), cudaMemcpyHostToDevice);
  CudaErrorCheck();
  cudaMemcpy(d_Growth_rate, Growth_rate, MaxNoofC180s*sizeof(float), cudaMemcpyHostToDevice);
  CudaErrorCheck();
  cudaMemcpy(d_CellINdex, CellINdex, MaxNoofC180s*sizeof(int), cudaMemcpyHostToDevice);
  CudaErrorCheck(); 
  cudaMemcpy(d_pressList, pressList, No_of_C180s*sizeof(float), cudaMemcpyHostToDevice);
  CudaErrorCheck();
  cudaMemcpy(d_ScaleFactor, ScaleFactor, No_of_C180s*sizeof(float), cudaMemcpyHostToDevice);
  CudaErrorCheck();
  cudaMemcpy(d_DivisionVolume, DivisionVolume, No_of_C180s*sizeof(float), cudaMemcpyHostToDevice);
  CudaErrorCheck();
  cudaMemcpy(d_gamma_env, gamma_env, No_of_C180s*sizeof(float), cudaMemcpyHostToDevice);
  CudaErrorCheck();
  cudaMemcpy(d_viscotic_damp, viscotic_damp, No_of_C180s*sizeof(float), cudaMemcpyHostToDevice);
  CudaErrorCheck();


/**************************************************************************************************************/


  // initialize device rng


    if (add_rands){
      curandGenerator_t gen;
      
      if (cudaMalloc((void **)&d_rngStates, sizeof(curandState)*192*MaxNoofC180s) != cudaSuccess){
          fprintf(stderr, "ERROR: Failed to allocate rng state memory in %s, at %d\n", __FILE__, __LINE__);
          return 1;
      }
          

      if (cudaMalloc((void **)&d_seeds, sizeof(unsigned int)*192*MaxNoofC180s) != cudaSuccess){
          fprintf(stderr, "ERROR: Failed to allocate rng seeds in %s, at %d\n", __FILE__, __LINE__);
          return 1;
      }
      
      time_t secs_since_1970;
      
      
      curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_MT19937);
      CudaErrorCheck();

      curandSetPseudoRandomGeneratorSeed(gen, time(&secs_since_1970));
      CudaErrorCheck();

      if ( rand_seed <= 0 ) printf("   rand_seed = %lu\n",(unsigned long)secs_since_1970);

      if (rand_seed > 0){
          curandSetPseudoRandomGeneratorSeed(gen, rand_seed);
          CudaErrorCheck();
      }

      curandGenerate(gen, d_seeds, MaxNoofC180s*192);
      CudaErrorCheck();
  
      DeviceRandInit<<<(192*MaxNoofC180s)/256 + 1, 256>>>(d_rngStates, d_seeds, 192*MaxNoofC180s);
      CudaErrorCheck();
  }


  noofblocks      = No_of_C180s;
  threadsperblock = 192;
  printf("   no of blocks = %d, threadsperblock = %d, no of threads = %ld\n",
         noofblocks, threadsperblock, ((long) noofblocks)*((long) threadsperblock));


  
  globalrank = 0;


  // open trajectory file
  
   if (Restart == 0){
  	trajfile = fopen (trajFileName, "w");
  }else{
  	trajfile = fopen (trajFileName, "a+");
  }
  
  if ( trajfile == NULL)
  {
      printf("Failed to open %s \n", trajFileName);
      return -1;
  }

  FILE* forceFile;
  if (Restart == 0){
    	 forceFile = fopen(forces_file, "w");
  }else{
   	 forceFile = fopen(forces_file, "a");
  }
  

  FILE* velFile;
  if (Restart == 0){
    	 velFile = fopen("velocity.xyz", "w");
  }else{
   	 velFile = fopen("velocity.xyz", "a+");
  }	


#ifdef OUTPUT_ADP_ERROR
  FILE* timeFile = fopen("times", "w");
  FILE* errFile = fopen("errors", "w"); 
#endif 


  


  float rGrowth = 0;
  bool growthDone = false;
  
  
  // Setup simulation box, if needed (non-pbc)
  if (useRigidSimulationBox){
      
      printf("   Setup rigid (non-PBC) box...\n"); 
 
      BoxCen.x = (boxMax.x - BoxMin.x)/2;
      BoxCen.y = (boxMax.y - BoxMin.y)/2;
      BoxCen.z = (boxMax.z - BoxMin.z)/2;
 
      
      if ((boxMax.z - BoxMin.z) < divVol){
      	//DL = divVol; 
      	DL = divisionV;
      } else {
      	DL = 1.4;
      }
      
      
      Xdiv = ceil((boxMax.x - BoxMin.x)/DL);
      printf (" %d \n",Xdiv);
      Ydiv = ceil((boxMax.y - BoxMin.y)/DL);
      printf (" %d \n",Ydiv);
      Zdiv = ceil((boxMax.z - BoxMin.z)/DL);
      printf (" %d \n",Zdiv); 

      printf("   Done!\n");
      printf("   Simulation box minima:\n   X: %f, Y: %f, Z: %f\n", BoxMin.x, BoxMin.y, BoxMin.z);
      printf("   Simulation box maximum:\n   X: %f, Y: %f, Z: %f\n", boxMax.x, boxMax.y, boxMax.z);
  }


  // Code to set up pbc things
  if (usePBCs || useLEbc){
    printf("   Setup PBC box...\n"); 
    
    BoxMin.x = 0.0;
    BoxMin.y = 0.0;
    BoxMin.z = 0.0;
    
    BoxCen.x = (boxMax.x - BoxMin.x)/2;
    BoxCen.y = (boxMax.y - BoxMin.y)/2;
    BoxCen.z = (boxMax.z - BoxMin.z)/2;
  

    if ((boxMax.z - BoxMin.z) < divVol){
      	//DL = divVol;
      	DL = divisionV; 
    } else {
      	DL = 1.4;
    }
    
    Xdiv = ceil((boxMax.x - BoxMin.x)/DL);
    DLp.x = (boxMax.x - BoxMin.x)/Xdiv;
    printf (" %d \n",Xdiv);
    Ydiv = ceil((boxMax.y - BoxMin.y)/DL);
    DLp.y = (boxMax.y - BoxMin.y)/Ydiv;
    printf (" %d \n",Ydiv);
    Zdiv = ceil((boxMax.z - BoxMin.y)/DL);
    DLp.z = (boxMax.z - BoxMin.y)/Zdiv;
    printf (" %d \n",Zdiv);  
    
    printf (" %f \n",DLp.x);
    printf (" %f \n",DLp.y);
    printf (" %f \n",DLp.z);

    printf("   Done!\n");
    printf("   Simulation box minima:\n   X: %f, Y: %f, Z: %f\n", BoxMin.x, BoxMin.y, BoxMin.z);
    printf("   Simulation box maximum:\n   X: %f, Y: %f, Z: %f\n", boxMax.x, boxMax.y, boxMax.z);
  }

  if (correct_com == true && !Restart){
  
      CenterOfMass<<<No_of_C180s,256>>>(No_of_C180s,
                                        d_X, d_Y, d_Z,
                                        d_CMx, d_CMy, d_CMz);

      CudaErrorCheck();
     
     
      reductionblocks = (No_of_C180s-1)/1024+1;
      SysCMpost<<<reductionblocks,1024>>> ( No_of_C180s, d_CMx, d_CMy, d_CMz, 
			   		      d_SysCx, d_SysCy, d_SysCz);
      CudaErrorCheck(); 


      SysCM<<<1,1024>>> (No_of_C180s, reductionblocks,
        		    d_SysCx, d_SysCy, d_SysCz,
			    d_sysCM);
      
         
      CudaErrorCheck();
      
      CorrectCoMMotion<<<(No_of_C180s*192)/1024 + 1, 1024>>>(d_X, d_Y, d_Z,
                                                             d_sysCM,BoxCen,
                                                             No_of_C180s*192);
      CudaErrorCheck();
      

      
  }
  

  if ( correct_Vcom == true && !Restart){
  
     
      VelocityCenterOfMass<<<No_of_C180s,256>>>(No_of_C180s,
                                        	  d_velListX, d_velListY, d_velListZ,
                                        	  d_VCMx, d_VCMy, d_VCMz);
      CudaErrorCheck();
      
      reductionblocks = (No_of_C180s-1)/1024+1;
      SysCMpost<<<reductionblocks,1024>>> ( No_of_C180s, d_VCMx, d_VCMy, d_VCMz, 
			   			d_SysCx, d_SysCy, d_SysCz);
      CudaErrorCheck(); 


      SysCM<<<1,1024>>> (No_of_C180s, reductionblocks,
        		    d_SysCx, d_SysCy, d_SysCz,
			    d_sysVCM);
      
     CudaErrorCheck();
      
      
      CorrectCoMVelocity<<<(No_of_C180s*192)/1024 + 1, 1024>>>(d_velListX, d_velListY, d_velListZ,
                                                               d_sysVCM,
                                                               No_of_C180s*192);
          
      CudaErrorCheck(); 
  }  
  


  CenterOfMass<<<No_of_C180s,256>>>(No_of_C180s, d_X, d_Y, d_Z, d_CMx, d_CMy, d_CMz);
  //DL = divVol; 
  CudaErrorCheck(); 

  if ( cudaSuccess != cudaMalloc( (void **)&d_NNlist ,    Xdiv*Ydiv*Zdiv*64*sizeof(int))) return(-1); 

  thrust::device_vector<int> d_NoofNNlistV(Xdiv*Ydiv*Zdiv);
  thrust::fill(d_NoofNNlistV.begin(), d_NoofNNlistV.end(), 0);
  int *d_NoofNNlist = thrust::raw_pointer_cast(&d_NoofNNlistV[0]);


  // Better way to see how much GPU memory is being used.
  size_t totalGPUMem;
  size_t freeGPUMem;

  if ( cudaSuccess != cudaMemGetInfo ( &freeGPUMem, &totalGPUMem ) ) {
      printf("Couldn't read GPU Memory status\nExiting...\n");
      CudaErrorCheck();
      exit(1);
  }

  GPUMemory = totalGPUMem - freeGPUMem;

 // Precalculate random plane
  initialize_Plane(MaxNoofC180s);


   if(usePBCs ){
        
       CoorUpdatePBC <<<No_of_C180s, threadsperblock>>> (d_X, d_Y, d_Z,
                                                          d_CMx, d_CMy, d_CMz,
                                                          boxMax, divVol, No_of_C180s,
                                                          useRigidBoxZ, useRigidBoxY, impurityNum);
  


       CudaErrorCheck();
   }
   if(useLEbc){
        
       UpdateLEbc <<<No_of_C180s, threadsperblock>>> (d_X, d_Y, d_Z,
                       				 d_velListX, d_velListY, d_velListZ, d_CMx, d_CMy, d_CMz,
                       				 boxMax, divVol, No_of_C180s, Pshift, Vshift, useRigidBoxZ, impurityNum);
                        
        CudaErrorCheck();	
	
    }





      if (useRigidSimulationBox){	
      		makeNNlist<<<No_of_C180s/512+1,512>>>( No_of_C180s, d_CMx, d_CMy, d_CMz,
        	Xdiv, Ydiv, Zdiv, BoxMin, d_NoofNNlist, d_NNlist, DL);
        
        	CudaErrorCheck(); 
       }
	if(usePBCs){
       	makeNNlistPBC<<<No_of_C180s/512+1,512>>>( No_of_C180s, d_CMx, d_CMy, d_CMz,
        	attraction_range, Xdiv, Ydiv, Zdiv, boxMax, d_NoofNNlist, d_NNlist, DLp, useRigidBoxZ,useRigidBoxY);
        
        	CudaErrorCheck(); 
       }
       if(useLEbc){
       	makeNNlistLEbc<<<No_of_C180s/512+1,512>>>( No_of_C180s, d_CMx, d_CMy, d_CMz,
        	attraction_range, Xdiv, Ydiv, Zdiv, boxMax, d_NoofNNlist, d_NNlist, DLp, Pshift, useRigidBoxZ);
        	
        	CudaErrorCheck();
       
       } 
  



  if (constrainAngles){


      d_theta0V = h_theta0; 
      CudaErrorCheck(); 
  }


  float r_CM_o = pow((3.0/4.0) * (1/3.14159) * divVol*2.0, 1.0/3);



  printf("   Total amount of GPU memory used =    %8.2lf MB\n",GPUMemory/(1024*1024.0));
  printf("   Total amount of CPU memory used =    %8.2lf MB\n",CPUMemory/(1024*1024.0));

  // initial conditions

  // X[10] *= 1.1;

  // cudaMemcpy(d_X, X, 192*No_of_C180s*sizeof(float), cudaMemcpyHostToDevice);

  if (useRigidSimulationBox){	
  	CalculateConForce<<<No_of_C180s,threadsperblock>>>(   No_of_C180s, d_C180_nn, d_C180_sign,
                                                     	d_X,  d_Y,  d_Z,
                                                     	d_CMx, d_CMy, d_CMz,
                                                     	d_R0, d_ScaleFactor, d_pressList, d_Youngs_mod, 
                                                     	attraction_strength, attraction_range,
                                                     	repulsion_strength, repulsion_range,
                                                     	d_viscotic_damp,
                                                     	Xdiv, Ydiv, Zdiv, boxMax,
                                                     	d_NoofNNlist, d_NNlist, DL, d_gamma_env,
                                                     	threshDist,
								BoxMin, Youngs_mod,
                                                     	constrainAngles, d_theta0, d_fConList, d_ExtForces,
                                                     	impurityNum,f_range); 
                                                     	
       CudaErrorCheck();
                                                     	
      CalculateDisForce<<<No_of_C180s, threadsperblock>>>(No_of_C180s, d_C180_nn, d_C180_sign, 
                                                        	d_X, d_Y, d_Z,
                                                        	d_CMx, d_CMy, d_CMz,
                                                        	internal_damping,
                                                        	attraction_range,
                                                        	d_viscotic_damp,
                                                        	Xdiv, Ydiv, Zdiv,BoxMin,
                                                        	d_NoofNNlist, d_NNlist, DL, d_gamma_env,
                                                        	d_velListX, d_velListY, d_velListZ,
                                                        	d_fDisList,impurityNum,f_range);
                                                        
                                                        
       CudaErrorCheck();                                                  
  }
  
  if(usePBCs){
  
    	CalculateConForcePBC<<<No_of_C180s,threadsperblock>>>( No_of_C180s, d_C180_nn, d_C180_sign,
                                                     	d_X,  d_Y,  d_Z,
                                                     	d_CMx, d_CMy, d_CMz,
                                                     	d_R0,d_ScaleFactor, d_pressList, d_Youngs_mod , 
                                                     	attraction_strength, attraction_range,
                                                     	repulsion_strength, repulsion_range,
                                                     	d_viscotic_damp,
                                                     	Xdiv, Ydiv, Zdiv, boxMax,
                                                     	d_NoofNNlist, d_NNlist, DLp, d_gamma_env,
                                                     	threshDist,
                                                     	BoxMin, Youngs_mod,
                                                     	constrainAngles, d_theta0, d_fConList,
                                                     	useRigidBoxZ,useRigidBoxY,impurityNum,f_range);
                                                     	
       CudaErrorCheck();                                             	
  	
  	                                                     	
      CalculateDisForcePBC<<<No_of_C180s, threadsperblock>>>(No_of_C180s, d_C180_nn, d_C180_sign, 
                                                        d_X, d_Y, d_Z,
                                                        d_CMx, d_CMy, d_CMz,
                                                        internal_damping,
                                                        attraction_range,
                                                        d_viscotic_damp,
                                                        Xdiv, Ydiv, Zdiv, boxMax,
                                                        d_NoofNNlist, d_NNlist, DLp, d_gamma_env,
                                                        d_velListX, d_velListY, d_velListZ,
                                                        d_fDisList, useRigidBoxZ,useRigidBoxY,impurityNum,f_range);
    CudaErrorCheck();	
  
  }
  if(useLEbc){
  
  CalculateConForceLEbc<<<No_of_C180s,threadsperblock>>>(No_of_C180s, d_C180_nn, d_C180_sign,
                                                     	d_X,  d_Y,  d_Z,
                                                     	d_CMx, d_CMy, d_CMz,
                                                     	d_R0,d_ScaleFactor, d_pressList, d_Youngs_mod , 
                                                     	attraction_strength, attraction_range,
                                                     	repulsion_strength, repulsion_range,
                                                     	d_viscotic_damp,
                                                     	Xdiv, Ydiv, Zdiv, boxMax,
                                                     	d_NoofNNlist, d_NNlist, DLp, d_gamma_env,
                                                     	threshDist,
                                                     	BoxMin, Youngs_mod,
                                                     	constrainAngles, d_theta0, d_fConList,
                                                     	Pshift,useRigidBoxZ,impurityNum,f_range);
                                                     	
       CudaErrorCheck();                                             	
  	
  	                                                     	
      CalculateDisForceLEbc<<<No_of_C180s, threadsperblock>>>(No_of_C180s, d_C180_nn, d_C180_sign, 
                                                        	d_X, d_Y, d_Z,
                                                        	d_CMx, d_CMy, d_CMz,
                                                        	internal_damping,
                                                        	attraction_range,
                                                       	d_viscotic_damp,
                                                        	Xdiv, Ydiv, Zdiv, boxMax,
                                                        	d_NoofNNlist, d_NNlist, DLp, d_gamma_env,
                                                        	d_velListX, d_velListY, d_velListZ,
                                                        	d_fDisList, Pshift, Vshift, useRigidBoxZ,impurityNum,f_range);
    CudaErrorCheck();	
  
  
  
  }
  
  	  volumes<<<No_of_C180s,192>>>(No_of_C180s, d_C180_56,
                                     d_X, d_Y, d_Z,
                                     d_CMx , d_CMy, d_CMz,
                                     d_volume, d_cell_div, d_DivisionVolume,
                                     checkSphericity, d_area, 
                                     stiffness1, useDifferentCell, d_Youngs_mod,
                                     recalc_r0, ApoVol ,d_cell_Apo, d_ScaleFactor);
  

	
  int t = MaxNoofC180s;	
  if (Restart ==0){
  
  	if (binaryOutput){
  
      		
      		fwrite(&t, sizeof(int), 1, trajfile);
      
      		t = (int)useDifferentCell;
      		fwrite(&t, sizeof(int), 1, trajfile);
      
      		t = (Time_steps+equiStepCount+1) / trajWriteInt;
      		fwrite(&t, sizeof(int), 1, trajfile);
      
    
     		WriteBinaryTraj(0, trajfile, 1); 
  	} else {
      		fprintf(trajfile, "Header Start:\n");
      		fprintf(trajfile, "Maximum number of cells:\n%d\n", MaxNoofC180s);

      		fprintf(trajfile, "Using variable stiffness:\n");
      		if (useDifferentCell) 
          		fprintf(trajfile, "True\n");
      		else
          		fprintf(trajfile, "False\n");

      		fprintf(trajfile, "Maximum number of frames:\n%d\n", (Time_steps+equiStepCount+1) / trajWriteInt);
     	 	fprintf(trajfile, "Header End\n");
      		write_traj(0, trajfile);
  	}
  	if (write_cont_force){
  
      		fprintf(forceFile, "step,num_cells,cell_ind,node_ind,glob_node_ind,FX,FY,FZ,F,FX_ext,FY_ext,FZ_ext,F_ext,VX,VY,VZ,V,X,Y,Z,P,Vol,Area\n");
      
      		cudaMemcpy(h_contactForces.x, d_fConList.x, 192*No_of_C180s*sizeof(float), cudaMemcpyDeviceToHost);
      		cudaMemcpy(h_contactForces.y, d_fConList.y, 192*No_of_C180s*sizeof(float), cudaMemcpyDeviceToHost);
      		cudaMemcpy(h_contactForces.z, d_fConList.z, 192*No_of_C180s*sizeof(float), cudaMemcpyDeviceToHost);
      		cudaMemcpy(h_ExtForces.x, d_ExtForces.x, 192*No_of_C180s*sizeof(float), cudaMemcpyDeviceToHost);
      		cudaMemcpy(h_ExtForces.y, d_ExtForces.y, 192*No_of_C180s*sizeof(float), cudaMemcpyDeviceToHost);
		cudaMemcpy(h_ExtForces.z, d_ExtForces.z, 192*No_of_C180s*sizeof(float), cudaMemcpyDeviceToHost);
      		
      		cudaMemcpy(velListX, d_velListX, 192*No_of_C180s*sizeof(float), cudaMemcpyDeviceToHost);
      		cudaMemcpy(velListY, d_velListY, 192*No_of_C180s*sizeof(float), cudaMemcpyDeviceToHost);
      		cudaMemcpy(velListZ, d_velListZ, 192*No_of_C180s*sizeof(float), cudaMemcpyDeviceToHost);

		cudaMemcpy(pressList, d_pressList, No_of_C180s*sizeof(float), cudaMemcpyDeviceToHost);      
      		cudaMemcpy(volume, d_volume, No_of_C180s*sizeof(float), cudaMemcpyDeviceToHost);
      		cudaMemcpy(area, d_area, No_of_C180s*sizeof(float), cudaMemcpyDeviceToHost);	
      		writeForces(forceFile, 0, No_of_C180s);
  	}
  	if(write_vel_file){
  	          
               cudaMemcpy(velListX, d_velListX, 192*No_of_C180s*sizeof(float), cudaMemcpyDeviceToHost);
               cudaMemcpy(velListY, d_velListY, 192*No_of_C180s*sizeof(float), cudaMemcpyDeviceToHost);
               cudaMemcpy(velListZ, d_velListZ, 192*No_of_C180s*sizeof(float), cudaMemcpyDeviceToHost);
               
               t = MaxNoofC180s;
               fwrite(&t, sizeof(int), 1, velFile);
      
      		t = (int)useDifferentCell;
      		fwrite(&t, sizeof(int), 1, velFile);
      
      		t = (Time_steps+equiStepCount+1) / trajWriteInt;
      		fwrite(&t, sizeof(int), 1, velFile);
               
               write_vel(0, velFile,1);
       }
  	
		  	
  	

  }

  //return 0;


  int numNodes = No_of_C180s*192;
  NewCellInd = No_of_C180s;
  WithoutApo = true;
  // Simulation loop
  for ( step = 1; step < Time_steps+1 + equiStepCount; step++)
  {
	Pshift = shear_rate*step*boxMax.x*delta_t;     



    
    //printf("step %d\n", step);
      numNodes = No_of_C180s*192;
      Integrate<<<No_of_C180s, threadsperblock>>>(d_X, d_Y, d_Z, 
                                                 d_velListX, d_velListY, d_velListZ, 
                                                 delta_t,  mass,
                                                 d_fConList, d_fDisList, d_fRanList,
                                                 No_of_C180s, impurityNum);
      CudaErrorCheck();


      //ForwardTime<<<No_of_C180s, threadsperblock>>>(d_XP, d_YP, d_ZP, 
      //                                             d_X , d_Y , d_Z ,
      //                                             No_of_C180s, impurityNum);
      //CudaErrorCheck();



// ------------------------------------------------------------------------------------------------

	if ( apoptosis && WithoutApo && No_of_C180s > popToStartApo && !colloidal_dynamics)
	{
	        printf("Cell apoptosis starts.\n");
		WithoutApo = false;
	}

	
	// ----------------------------------------- Begin Cell Death ------------	
	if (apoptosis && !WithoutApo) {	
     
		
      		float rans[1];
        	int DInd;
        	int Aporank;
      		
      		
      		if ((step)%1000 == 0){
      		
            		
            		cudaMemcpy(Growth_rate, d_Growth_rate, No_of_C180s*sizeof(float), cudaMemcpyDeviceToHost);
            		CudaErrorCheck(); 
            		
                       cudaMemcpy(CellINdex, d_CellINdex, No_of_C180s*sizeof(int), cudaMemcpyDeviceToHost);
        	        CudaErrorCheck();
	 		      		
     		 	for (int s= impurityNum; s < No_of_C180s - impurityNum; s++){
      		 		  	 	
 	
      		 		if (CellINdex[s] >= 0){
      		 		
      		 			if (Growth_rate[s] == - squeeze_rate) continue; 		
      		 		
        	        		ranmar(rans, 1);
        	        		if (rans[0] < Apo_rate*0.1){
        	        			Growth_rate[s] = - squeeze_rate;
        	        			NumApoCell ++;
        	        		}
        	        	
        	        	} else {
        	        		
        	        	       	
        	        	
        	        	       if (Growth_rate[DInd] == - squeeze_rate2 || !useDifferentCell) continue;	
      		 			
        	        		ranmar(rans, 1);
        	        		if(rans[0] < Apo_rate2*0.1){
        	        			Growth_rate[DInd] = - squeeze_rate2;
        	        			NumApoCell ++;	
        	        		}
        	        	
        	        	
        	        	
        	        	
        	        	}			
        	        	
        	        }

        	        
        	        cudaMemcpy(d_Growth_rate, Growth_rate, No_of_C180s*sizeof(float), cudaMemcpyHostToDevice);
            		CudaErrorCheck();      	        
        	        
			count_and_die();
			
			if (num_cell_Apo> 0){
			
			        cudaMemcpy(X , d_X, No_of_C180s*192*sizeof(float), cudaMemcpyDeviceToHost);
  				CudaErrorCheck();
  				cudaMemcpy(Y , d_Y, No_of_C180s*192*sizeof(float), cudaMemcpyDeviceToHost);
  				CudaErrorCheck();
  				cudaMemcpy(Z , d_Z, No_of_C180s*192*sizeof(float), cudaMemcpyDeviceToHost);
  				CudaErrorCheck();
  				cudaMemcpy(velListX , d_velListX , No_of_C180s*192*sizeof(float), cudaMemcpyDeviceToHost);
  				CudaErrorCheck();
  				cudaMemcpy(velListY , d_velListY , No_of_C180s*192*sizeof(float), cudaMemcpyDeviceToHost);
  				CudaErrorCheck();
  				cudaMemcpy(velListZ , d_velListZ , No_of_C180s*192*sizeof(float), cudaMemcpyDeviceToHost);
  				CudaErrorCheck();
        	               cudaMemcpy(pressList, d_pressList, No_of_C180s*sizeof(float), cudaMemcpyDeviceToHost);
        	               CudaErrorCheck();
        	               cudaMemcpy(youngsModArray, d_Youngs_mod ,No_of_C180s*sizeof(float), cudaMemcpyDeviceToHost);
  				CudaErrorCheck();
  				cudaMemcpy(Growth_rate, d_Growth_rate, No_of_C180s*sizeof(float), cudaMemcpyDeviceToHost);
        	               CudaErrorCheck();			
			        cudaMemcpy(ScaleFactor, d_ScaleFactor, sizeof(float)*No_of_C180s, cudaMemcpyDeviceToHost);
            			CudaErrorCheck();
            			cudaMemcpy(DivisionVolume, d_DivisionVolume, sizeof(float)*No_of_C180s, cudaMemcpyDeviceToHost);
            			CudaErrorCheck();
            			cudaMemcpy(gamma_env, d_gamma_env, sizeof(float)*No_of_C180s, cudaMemcpyDeviceToHost);
            			CudaErrorCheck();
            			cudaMemcpy(viscotic_damp, d_viscotic_damp, sizeof(float)*No_of_C180s, cudaMemcpyDeviceToHost);
           			CudaErrorCheck();
			        cudaMemcpy(CellINdex, d_CellINdex, No_of_C180s*sizeof(int), cudaMemcpyDeviceToHost);
        	               CudaErrorCheck();

				size_t EndShift;
				for (int ApoCell = 0; ApoCell < num_cell_Apo; ApoCell++) {
          			
          			
          				Aporank = cell_Apo_inds[ApoCell] - ApoCell; 
          				EndShift =  (No_of_C180s - Aporank-1)*192*sizeof(float); 


  				
  					cudaMemcpy(X + 192*Aporank,  d_X + 192*(Aporank + 1), EndShift ,cudaMemcpyDeviceToHost);
  					CudaErrorCheck();
  					cudaMemcpy(Y + 192*Aporank,  d_Y + 192*(Aporank + 1), EndShift ,cudaMemcpyDeviceToHost);
  					CudaErrorCheck();
  					cudaMemcpy(Z + 192*Aporank,  d_Z + 192*(Aporank + 1), EndShift ,cudaMemcpyDeviceToHost);
  					CudaErrorCheck();	                  	                  
   	     	               	cudaMemcpy(velListX + 192*Aporank, d_velListX + 192*(Aporank + 1), EndShift , cudaMemcpyDeviceToHost);
               			CudaErrorCheck();
               			cudaMemcpy(velListY + 192*Aporank, d_velListY + 192*(Aporank + 1), EndShift , cudaMemcpyDeviceToHost);
               			CudaErrorCheck();
               			cudaMemcpy(velListZ + 192*Aporank, d_velListZ + 192*(Aporank + 1), EndShift , cudaMemcpyDeviceToHost);
        	               	CudaErrorCheck();	                  
        	               	cudaMemcpy(pressList + Aporank, d_pressList + Aporank+1, (MaxNoofC180s - Aporank - 1)*sizeof(float), cudaMemcpyDeviceToHost);
        	               	CudaErrorCheck();
        	               	cudaMemcpy(Growth_rate + Aporank, d_Growth_rate + Aporank+1, (MaxNoofC180s - Aporank - 1)*sizeof(float), cudaMemcpyDeviceToHost);
        	               	CudaErrorCheck();
        	               	cudaMemcpy(youngsModArray + Aporank, d_Youngs_mod + Aporank+1, (MaxNoofC180s - Aporank - 1)*sizeof(float), cudaMemcpyDeviceToHost);
  					CudaErrorCheck();
        	               	cudaMemcpy(ScaleFactor + Aporank, d_ScaleFactor + Aporank+1, (MaxNoofC180s - Aporank - 1)*sizeof(float), cudaMemcpyDeviceToHost);
  					CudaErrorCheck();
        	               	cudaMemcpy(DivisionVolume + Aporank, d_DivisionVolume + Aporank+1, (MaxNoofC180s - Aporank - 1)*sizeof(float), cudaMemcpyDeviceToHost);
  					CudaErrorCheck();  					  					
        	               	cudaMemcpy(gamma_env + Aporank, d_gamma_env + Aporank+1, (MaxNoofC180s - Aporank - 1)*sizeof(float), cudaMemcpyDeviceToHost);
  					CudaErrorCheck();  					
        	               	cudaMemcpy(viscotic_damp + Aporank, d_viscotic_damp + Aporank+1, (MaxNoofC180s - Aporank - 1)*sizeof(float), cudaMemcpyDeviceToHost);
  					CudaErrorCheck();			
        	               	cudaMemcpy(CellINdex + Aporank, d_CellINdex + Aporank+1, (MaxNoofC180s - Aporank - 1)*sizeof(int), cudaMemcpyDeviceToHost);
        	               	CudaErrorCheck();

      				
      					-- No_of_C180s;
      				
      							
      					cudaMemcpy(d_X , X, No_of_C180s*192*sizeof(float), cudaMemcpyHostToDevice);
  					CudaErrorCheck();
  					cudaMemcpy(d_Y , Y, No_of_C180s*192*sizeof(float), cudaMemcpyHostToDevice);
  					CudaErrorCheck();
  					cudaMemcpy(d_Z , Z, No_of_C180s*192*sizeof(float), cudaMemcpyHostToDevice);
  					CudaErrorCheck();
  					cudaMemcpy(d_velListX , velListX , No_of_C180s*192*sizeof(float), cudaMemcpyHostToDevice);
  					CudaErrorCheck();
  					cudaMemcpy(d_velListY , velListY , No_of_C180s*192*sizeof(float), cudaMemcpyHostToDevice);
  					CudaErrorCheck();
  					cudaMemcpy(d_velListZ , velListZ , No_of_C180s*192*sizeof(float), cudaMemcpyHostToDevice);
  					CudaErrorCheck();
        	               	cudaMemcpy(d_pressList, pressList, No_of_C180s*sizeof(float), cudaMemcpyHostToDevice);
        	               	CudaErrorCheck();
  					cudaMemcpy(d_Growth_rate, Growth_rate, No_of_C180s*sizeof(float), cudaMemcpyHostToDevice);
        	               	CudaErrorCheck();
        	               	cudaMemcpy(d_Youngs_mod, youngsModArray ,No_of_C180s*sizeof(float), cudaMemcpyHostToDevice );
  					CudaErrorCheck();
        	               	cudaMemcpy(d_ScaleFactor, ScaleFactor ,No_of_C180s*sizeof(float), cudaMemcpyHostToDevice );
  					CudaErrorCheck();  					
        	               	cudaMemcpy(d_DivisionVolume, DivisionVolume ,No_of_C180s*sizeof(float), cudaMemcpyHostToDevice );
  					CudaErrorCheck();  					
        	               	cudaMemcpy(d_gamma_env, gamma_env ,No_of_C180s*sizeof(float), cudaMemcpyHostToDevice );
  					CudaErrorCheck();  					
        	               	cudaMemcpy(d_viscotic_damp, viscotic_damp ,No_of_C180s*sizeof(float), cudaMemcpyHostToDevice );
  					CudaErrorCheck();
        	               	cudaMemcpy(d_CellINdex, CellINdex, No_of_C180s*sizeof(int), cudaMemcpyHostToDevice);
        	               	CudaErrorCheck();  				

      					
      					NumRemoveCell ++;

      				}
      			}       
      		}	

	}
// ----------------------------------------- End Cell Death --------------



      CenterOfMass<<<No_of_C180s,256>>>(No_of_C180s, d_X, d_Y, d_Z, d_CMx, d_CMy, d_CMz);
      //DL = divVol; 
      CudaErrorCheck(); 

      cudaMemset(d_NoofNNlist, 0, Xdiv*Ydiv*Zdiv*sizeof(int));

      if (useRigidSimulationBox){	
      		makeNNlist<<<No_of_C180s/512+1,512>>>( No_of_C180s, d_CMx, d_CMy, d_CMz,
        	Xdiv, Ydiv, Zdiv, BoxMin, d_NoofNNlist, d_NNlist, DL);
        
        	CudaErrorCheck(); 
       }
	if(usePBCs){
       	makeNNlistPBC<<<No_of_C180s/512+1,512>>>( No_of_C180s, d_CMx, d_CMy, d_CMz,
        	attraction_range, Xdiv, Ydiv, Zdiv, boxMax, d_NoofNNlist, d_NNlist, DLp, useRigidBoxZ,useRigidBoxY);
        
        	CudaErrorCheck(); 
       }
       if(useLEbc){
       	makeNNlistLEbc<<<No_of_C180s/512+1,512>>>( No_of_C180s, d_CMx, d_CMy, d_CMz,
        	attraction_range, Xdiv, Ydiv, Zdiv, boxMax, d_NoofNNlist, d_NNlist, DLp, Pshift, useRigidBoxZ);
        	
        	CudaErrorCheck();
       
       }


// ---------------------------------------------------------------------------------------------------




      // save previous step forces in g
      
      if (doPopModel == 1){
            rGrowth = rMax * (1 - (No_of_C180s*1.0/maxPop));
            // dr = -rGrowth(a + b*rGrowth)
            // rGrowth += dr * delta_t ;
            // dN/dT = N*R
            // dR/dT = -R(a+bR)
            // 
            if (rGrowth < 0) rGrowth =0; 
      }
      else {
      		rGrowth = rMax;
      }

      if (!colloidal_dynamics){      
      	PressureUpdate <<<No_of_C180s/1024 + 1, 1024>>> (d_pressList, maxPressure, d_Growth_rate, No_of_C180s,
        	                                           d_Youngs_mod, impurityNum);
      		CudaErrorCheck(); 
      }
      
      if ( (step)%1000 == 0)
      {
          printf("   time %-8d %d cells, CellInApoptosis %d, NumCellDeath %d\n",step,No_of_C180s, NumApoCell, NumRemoveCell);
      }


#ifdef FORCE_DEBUG
      printf("time %d  pressure = %f\n", step, Pressure);
#endif

 if (useRigidSimulationBox){	
  	CalculateConForce<<<No_of_C180s,threadsperblock>>>(   No_of_C180s, d_C180_nn, d_C180_sign,
                                                     	d_X,  d_Y,  d_Z,
                                                     	d_CMx, d_CMy, d_CMz,
                                                     	d_R0, d_ScaleFactor, d_pressList, d_Youngs_mod, 
                                                     	attraction_strength, attraction_range,
                                                     	repulsion_strength, repulsion_range,
                                                     	d_viscotic_damp,
                                                     	Xdiv, Ydiv, Zdiv, boxMax,
                                                     	d_NoofNNlist, d_NNlist, DL, d_gamma_env,
                                                     	threshDist,
								BoxMin, Youngs_mod,
                                                     	constrainAngles, d_theta0, d_fConList, d_ExtForces,
                                                     	impurityNum,f_range); 
                                                     	
       CudaErrorCheck();
                                                     	
      CalculateDisForce<<<No_of_C180s, threadsperblock>>>(No_of_C180s, d_C180_nn, d_C180_sign, 
                                                        	d_X, d_Y, d_Z,
                                                        	d_CMx, d_CMy, d_CMz,
                                                        	internal_damping,
                                                        	attraction_range,
                                                        	d_viscotic_damp,
                                                        	Xdiv, Ydiv, Zdiv,BoxMin,
                                                        	d_NoofNNlist, d_NNlist, DL, d_gamma_env,
                                                        	d_velListX, d_velListY, d_velListZ,
                                                        	d_fDisList,impurityNum,f_range);
                                                        
                                                        
       CudaErrorCheck();                                                  
  }
  
  if(usePBCs){
  
    	CalculateConForcePBC<<<No_of_C180s,threadsperblock>>>( No_of_C180s, d_C180_nn, d_C180_sign,
                                                     	d_X,  d_Y,  d_Z,
                                                     	d_CMx, d_CMy, d_CMz,
                                                     	d_R0,d_ScaleFactor, d_pressList, d_Youngs_mod , 
                                                     	attraction_strength, attraction_range,
                                                     	repulsion_strength, repulsion_range,
                                                     	d_viscotic_damp,
                                                     	Xdiv, Ydiv, Zdiv, boxMax,
                                                     	d_NoofNNlist, d_NNlist, DLp, d_gamma_env,
                                                     	threshDist,
                                                     	BoxMin, Youngs_mod,
                                                     	constrainAngles, d_theta0, d_fConList,
                                                     	useRigidBoxZ,useRigidBoxY,impurityNum,f_range);
                                                     	
       CudaErrorCheck();                                             	
  	
  	                                                     	
      CalculateDisForcePBC<<<No_of_C180s, threadsperblock>>>(No_of_C180s, d_C180_nn, d_C180_sign, 
                                                        d_X, d_Y, d_Z,
                                                        d_CMx, d_CMy, d_CMz,
                                                        internal_damping,
                                                        attraction_range,
                                                        d_viscotic_damp,
                                                        Xdiv, Ydiv, Zdiv,boxMax,
                                                        d_NoofNNlist, d_NNlist, DLp, d_gamma_env,
                                                        d_velListX, d_velListY, d_velListZ,
                                                        d_fDisList, useRigidBoxZ,useRigidBoxY,impurityNum,f_range);
    CudaErrorCheck();	
  
  }
  if(useLEbc){
  
  CalculateConForceLEbc<<<No_of_C180s,threadsperblock>>>( No_of_C180s, d_C180_nn, d_C180_sign,
                                                     	d_X,  d_Y,  d_Z,
                                                     	d_CMx, d_CMy, d_CMz,
                                                     	d_R0,d_ScaleFactor, d_pressList, d_Youngs_mod , 
                                                     	attraction_strength, attraction_range,
                                                     	repulsion_strength, repulsion_range,
                                                     	d_viscotic_damp,
                                                     	Xdiv, Ydiv, Zdiv, boxMax,
                                                     	d_NoofNNlist, d_NNlist, DLp, d_gamma_env,
                                                     	threshDist,
                                                     	BoxMin, Youngs_mod,
                                                     	constrainAngles, d_theta0, d_fConList,
                                                     	Pshift,useRigidBoxZ,impurityNum,f_range);
                                                     	
       CudaErrorCheck();                                             	
  	
  	                                                     	
      CalculateDisForceLEbc<<<No_of_C180s, threadsperblock>>>(No_of_C180s, d_C180_nn, d_C180_sign, 
                                                        d_X, d_Y, d_Z,
                                                        d_CMx, d_CMy, d_CMz,
                                                        internal_damping,
                                                        attraction_range,
                                                        d_viscotic_damp,
                                                        Xdiv, Ydiv, Zdiv,boxMax,
                                                        d_NoofNNlist, d_NNlist, DLp, d_gamma_env,
                                                        d_velListX, d_velListY, d_velListZ,
                                                        d_fDisList, Pshift, Vshift, useRigidBoxZ,impurityNum,f_range);
    CudaErrorCheck();	
  
  
  
  }
       

      // Calculate random Force here...
      if (add_rands){
          CalculateRanForce<<<No_of_C180s, threadsperblock>>>(No_of_C180s, d_rngStates, rand_scale_factor,
                                                              d_fRanList, impurityNum);
          CudaErrorCheck();
      }
      
      VelocityUpdateA<<<No_of_C180s, threadsperblock>>>(d_velListX, d_velListY, d_velListZ,
                                                        d_fConList, d_fRanList, delta_t, numNodes, mass, impurityNum);
      CudaErrorCheck();


      // Dissipative velocity update part...
      for (int s = 0; s < 1; ++s){
      
          VelocityUpdateB<<<No_of_C180s, threadsperblock>>>(d_velListX, d_velListY, d_velListZ,
                                                           d_fDisList, delta_t, numNodes, mass, impurityNum);
          CudaErrorCheck();
          
          if (useRigidSimulationBox){	
                                                     	
     		CalculateDisForce<<<No_of_C180s, threadsperblock>>>(No_of_C180s, d_C180_nn, d_C180_sign, 
                                                        d_X, d_Y, d_Z,
                                                        d_CMx, d_CMy, d_CMz,
                                                        internal_damping,
                                                        attraction_range,
                                                        d_viscotic_damp,
                                                        Xdiv, Ydiv, Zdiv, BoxMin,
                                                        d_NoofNNlist, d_NNlist, DL, d_gamma_env,
                                                        d_velListX, d_velListY, d_velListZ,
                                                        d_fDisList, impurityNum,f_range);
                                                        
       	CudaErrorCheck();                                                  
  	}
  	if(usePBCs){              	
      		CalculateDisForcePBC<<<No_of_C180s, threadsperblock>>>(No_of_C180s, d_C180_nn, d_C180_sign, 
                	                                        d_X, d_Y, d_Z,
                	                                        d_CMx, d_CMy, d_CMz,
                	                                        internal_damping,
                	                                        attraction_range,
                	                                        d_viscotic_damp,
                	                                        Xdiv, Ydiv, Zdiv,boxMax,
                	                                        d_NoofNNlist, d_NNlist, DLp, d_gamma_env,
                	                                        d_velListX, d_velListY, d_velListZ,
                	                                        d_fDisList,useRigidBoxZ,useRigidBoxY, impurityNum,f_range);
    		CudaErrorCheck();	
  
  	}
  	if(useLEbc){
  	
  	  	                                                     	
      		CalculateDisForceLEbc<<<No_of_C180s, threadsperblock>>>(No_of_C180s, d_C180_nn, d_C180_sign, 
               	                                         d_X, d_Y, d_Z,
               	                                         d_CMx, d_CMy, d_CMz,
               	                                         internal_damping,
               	                                         attraction_range,
               	                                         d_viscotic_damp,
               	                                         Xdiv, Ydiv, Zdiv,boxMax,
               	                                         d_NoofNNlist, d_NNlist, DLp, d_gamma_env,
               	                                         d_velListX, d_velListY, d_velListZ,
               	                                         d_fDisList, Pshift, Vshift, useRigidBoxZ, impurityNum,f_range);
    CudaErrorCheck();
  	
  	}
  	
     }


      
      CenterOfMass<<<No_of_C180s,256>>>(No_of_C180s,d_X, d_Y, d_Z,
                                        d_CMx, d_CMy, d_CMz);

        CudaErrorCheck();



      if (step <= Time_steps && rGrowth > 0 && !colloidal_dynamics){
        // ------------------------------ Begin Cell Division ------------------------------------------------


        volumes<<<No_of_C180s,192>>>(No_of_C180s, d_C180_56,
                                     d_X, d_Y, d_Z,
                                     d_CMx , d_CMy, d_CMz,
                                     d_volume, d_cell_div, d_DivisionVolume,
                                     checkSphericity, d_area,
                                     stiffness1, useDifferentCell, d_Youngs_mod,
                                     recalc_r0,ApoVol,d_cell_Apo, d_ScaleFactor);
        CudaErrorCheck();



        count_and_get_div();

	//cudaStream_t *streams = (cudaStream_t *)malloc(num_cell_div*sizeof(cudaStream_t));
	
	//for (int i = 0 ; i < num_cell_div; i++) cudaStreamCreate(&streams[i]);
	//CudaErrorCheck();
	
        
        for (int divCell = 0; divCell < num_cell_div; divCell++) {
          
          globalrank = cell_div_inds[divCell];
   

          cell_division<<<1,256>>>(globalrank, 
                                   d_X, d_Y, d_Z, 
                                   d_CMx, d_CMy, d_CMz,
                                   d_velListX, d_velListY, d_velListZ,
                                   No_of_C180s, repulsion_range, d_asym,
                                   useDifferentCell, daughtSame,
                                   NewCellInd, stiffness1, rMax, divVol, gamma_visc, viscotic_damping,
                                   d_ScaleFactor, d_Youngs_mod, d_Growth_rate, d_DivisionVolume,
                                   d_gamma_env, d_viscotic_damp, d_CellINdex,
                                   d_DivPlane);
                                   
          CudaErrorCheck()
          
          resetIndices[divCell] = globalrank;
          resetIndices[divCell + num_cell_div] = No_of_C180s;
          
          
          ++No_of_C180s;
          ++NewCellInd;
          
          
          if (No_of_C180s > MaxNoofC180s){
          
              printf("ERROR: Population is %d, only allocated enough memory for %d\n",
                     No_of_C180s, MaxNoofC180s);
              printf("ERROR: Fatal error, crashing...\n");
              return -69;
          }
        
         
        }
        
        
        //for (int i = 0 ; i < num_cell_div; i++) cudaStreamDestroy(streams[i]);
	//CudaErrorCheck();
        
        if (num_cell_div>0){
            
            cudaMemcpy(d_resetIndices, resetIndices, 2*num_cell_div*sizeof(int), cudaMemcpyHostToDevice);
            CudaErrorCheck(); 

            PressureReset <<<(2*num_cell_div)/512 + 1, 512>>> (d_resetIndices, d_pressList, minPressure, 2*num_cell_div); 
            CudaErrorCheck();	 		      	            

        }


	if (countCells) {

        	if (countOnlyInternal == 1){
        
         		CenterOfMass<<<No_of_C180s,256>>>(No_of_C180s,
               		                         d_X, d_Y, d_Z,
               	        	                 d_CMx, d_CMy, d_CMz);

      			CudaErrorCheck();
     
     
      			reductionblocks = (No_of_C180s-1)/1024+1;
      			SysCMpost<<<reductionblocks,1024>>> ( No_of_C180s, d_CMx, d_CMy, d_CMz, 
		    		   		          d_SysCx, d_SysCy, d_SysCz);
      			CudaErrorCheck(); 


      			SysCM<<<1,1024>>> (No_of_C180s, reductionblocks,
        		  		      d_SysCx, d_SysCy, d_SysCz,
					      d_sysCM);
      
         
      			CudaErrorCheck();
      			
      			cudaMemcpy(h_sysCM.x, d_sysCM.x, sizeof(float), cudaMemcpyHostToDevice);
      			cudaMemcpy(h_sysCM.y, d_sysCM.y, sizeof(float), cudaMemcpyHostToDevice);
      			cudaMemcpy(h_sysCM.z, d_sysCM.z, sizeof(float), cudaMemcpyHostToDevice);
                       CudaErrorCheck();

//			h_SCM = d_SCM;
        
        
          		num_cell_div -= num_cells_far();
          
        	}

        	num_new_cells_per_step[step-1] = num_cell_div;
        	
        	if (step%newCellCountInt == 0){
          		newcells = 0;
          		for (int i = 0; i < newCellCountInt; i++) {
            			newcells += num_new_cells_per_step[countOffset + i];
          		}
          		dividingCells[(step-1)/newCellCountInt] = newcells;
          		totalCells[(step-1)/newCellCountInt] = No_of_C180s - newcells;
          		// Need to make sure this is how MIs are even calculated
          		countOffset += newCellCountInt;
        	}
      
       }
      
      
#if defined(FORCE_DEBUG) || defined(PRINT_VOLUMES)
      
      if (checkSphericity){
          //cudaMemcpy(volume, d_volume, No_of_C180s*sizeof(float), cudaMemcpyDeviceToHost);
          h_volume = d_volumeV; 
          cudaMemcpy(area, d_area, No_of_C180s*sizeof(float), cudaMemcpyDeviceToHost);
          cudaMemcpy(pressList, d_pressList, No_of_C180s*sizeof(float), cudaMemcpyDeviceToHost);
          printf("time: %d\n", step); 
          for (int i = 0; i < No_of_C180s; i++){
              printf ("Cell: %d, volume= %f, area=%f, psi=%f, p = %f" , i, h_volume[i], area[i],
                      4.835975862049408*pow(h_volume[i], 2.0/3.0)/area[i], pressList[i]);
          
              if (h_volume[i] > divVol)
                  printf(", I'm too big :(");
          
              printf("\n"); 
          }
      } else{
          h_volume = d_volumeV; 
          for (int i = 0; i < No_of_C180s; i++){
              printf ("Cell: %d, volume= %f", i, h_volume[i]); 
          
              if (h_volume[i] > divVol)
                  printf(", I'm too big :(");
          
              printf("\n"); 
          }
      }
#endif
      
      
        // --------------------------------------- End Cell Division -----------
      }


      CenterOfMass<<<No_of_C180s,256>>>(No_of_C180s, d_X, d_Y, d_Z, d_CMx, d_CMy, d_CMz);
      //DL = divVol; 
      CudaErrorCheck(); 



   if(usePBCs && (step)%2000 == 0){
        
            CoorUpdatePBC <<<No_of_C180s, threadsperblock>>> (d_X, d_Y, d_Z,
                                                              d_CMx, d_CMy, d_CMz,
                                                              boxMax, divVol, No_of_C180s,
                                                              useRigidBoxZ, useRigidBoxY, impurityNum);
  

		
            CudaErrorCheck();
        }
        
        if(useLEbc && step%1000 == 0){
       
            UpdateLEbc <<<No_of_C180s, threadsperblock>>> (d_X, d_Y, d_Z,
                        					d_velListX, d_velListY, d_velListZ, d_CMx, d_CMy, d_CMz,
                        					boxMax, divVol, No_of_C180s, Pshift, Vshift, useRigidBoxZ, impurityNum);
	
	
	}



      if (!growthDone && step > Time_steps+1){
          printf("Cell growth halted.\nProceeding with MD simulation without growth...\n");
          growthDone = true;
          
          if (useDifferentCell && !duringGrowth && !colloidal_dynamics){
              printf("Now making some cells softer...\n");
              int softCellCounter = 0;
              if (fractionOfCells > 0){
                  numberOfCells = roundf(fractionOfCells*No_of_C180s); 
              }

              printf("Will make %d cells softer\n", numberOfCells); 
              
              if (chooseRandomCellIndices){
                  float rnd[1];
                  //int* chosenIndices = (int*)malloc(numberOfCells, sizeof(int));
                  int chosenIndices[numberOfCells]; 
                  
                  for (int i = 0; i < numberOfCells; i++){
                      chosenIndices[i] = -1; 
                  }
                  
                  bool indexChosen = false;
                  int cellInd = -1;

                  printf("Make cells with indices "); 
                  
                  while (softCellCounter < numberOfCells){
                      ranmar(rnd, 1);
                      cellInd = roundf(rnd[0] * No_of_C180s);

                      for (int i = 0; i < softCellCounter; i++){
                          if (chosenIndices[i] == cellInd){
                              indexChosen = true;
                              break;
                          }
                      }

                      if (!indexChosen){
                          chosenIndices[softCellCounter] = cellInd;
                          softCellCounter++;
                          printf("%d ", cellInd); 
                      } else
                          indexChosen = false;
                      
                  }

                  //free(chosenIndices);

                  for (int i = 0; i < numberOfCells; i++){
                      youngsModArray[chosenIndices[i]] = Stiffness2; 
                  }
              }
              else {
                  // search for the oldest cells near the center of the system, and make them soft

      		   CenterOfMass<<<No_of_C180s,256>>>(No_of_C180s,
               	                         d_X, d_Y, d_Z,
                       	                 d_CMx, d_CMy, d_CMz);

      		   CudaErrorCheck();
     
     
      		   reductionblocks = (No_of_C180s-1)/1024+1;
      		   SysCMpost<<<reductionblocks,1024>>> ( No_of_C180s, d_CMx, d_CMy, d_CMz, 
		    		   		          d_SysCx, d_SysCy, d_SysCz);
      		   CudaErrorCheck(); 


      		   SysCM<<<1,1024>>> (No_of_C180s, reductionblocks,
        	  		      d_SysCx, d_SysCy, d_SysCz,
				      d_sysCM);
      
         
      		   CudaErrorCheck();


      		   cudaMemcpy(h_sysCM.x, d_sysCM.x, sizeof(float), cudaMemcpyHostToDevice);
      		   cudaMemcpy(h_sysCM.y, d_sysCM.y, sizeof(float), cudaMemcpyHostToDevice);
      		   cudaMemcpy(h_sysCM.z, d_sysCM.z, sizeof(float), cudaMemcpyHostToDevice);
                   CudaErrorCheck();


//		   h_SCM = d_SCM;	
                  
                  float Rmax2 = getRmax2();
                  float R2, dx, dy, dz;
                  int cellInd = 0; 


                  float f = 1 - closenessToCenter;
              
                  printf("Made cells with indices "); 

                  while (softCellCounter < numberOfCells && cellInd < No_of_C180s){
                      dx = CMx[cellInd] - *h_sysCM.x; 
                      dy = CMy[cellInd] - *h_sysCM.y; 
                      dz = CMz[cellInd] - *h_sysCM.z;

                      R2 = dx*dx + dy*dy + dz*dz;

                      if (R2 <= f*f*Rmax2){
                          printf("%d, ", cellInd); 
                          softCellCounter++; 
                          youngsModArray[cellInd] = Stiffness2; 

                      }
                      cellInd++; 
                  }
              }
              
              cudaMemcpy(d_Youngs_mod, youngsModArray, No_of_C180s*sizeof(float), cudaMemcpyHostToDevice);
              printf("\b\b softer\n"); 
          }

      }

      if (correct_com == true){


      		CenterOfMass<<<No_of_C180s,256>>>(No_of_C180s,
               	                         d_X, d_Y, d_Z,
                       	                 d_CMx, d_CMy, d_CMz);

      		CudaErrorCheck();
     
     
      		reductionblocks = (No_of_C180s-1)/1024+1;
      		SysCMpost<<<reductionblocks,1024>>> ( No_of_C180s, d_CMx, d_CMy, d_CMz, 
				   		      d_SysCx, d_SysCy, d_SysCz);
      		CudaErrorCheck(); 


      		SysCM<<<1,1024>>> (No_of_C180s, reductionblocks,
        			    d_SysCx, d_SysCy, d_SysCz,
				    d_sysCM);
      
         
      		CudaErrorCheck();
      
      		CorrectCoMMotion<<<(No_of_C180s*192)/1024 + 1, 1024>>>(d_X, d_Y, d_Z,
               	                                              d_sysCM, BoxCen,
               	                                              No_of_C180s*192);
      		CudaErrorCheck();
      
      		//cudaMemcpy(h_sysCM.x, d_sysCM.x, sizeof(float), cudaMemcpyHostToDevice);
      		//cudaMemcpy(h_sysCM.y, d_sysCM.y, sizeof(float), cudaMemcpyHostToDevice);
      		//cudaMemcpy(h_sysCM.z, d_sysCM.z, sizeof(float), cudaMemcpyHostToDevice);
                //CudaErrorCheck();
      		//printf("sysCMx = 	%f, sysCMy = 		%f, sysCmz = 		%f\n", h_sysCM.x, h_sysCM.y, h_sysCM.z);
      
      }

    if ( correct_Vcom == true ){
     
     
       VelocityCenterOfMass<<<No_of_C180s,256>>>(No_of_C180s,
                                          	    d_velListX, d_velListY, d_velListZ,
                                          	    d_VCMx, d_VCMy, d_VCMz);
                                          
        CudaErrorCheck();                        
                  
        reductionblocks = (No_of_C180s-1)/1024+1;
        
        SysCMpost<<<reductionblocks,1024>>> ( No_of_C180s, d_VCMx, d_VCMy, d_VCMz, 
			   			d_SysCx, d_SysCy, d_SysCz);
        CudaErrorCheck(); 


        SysCM<<<1,1024>>> (No_of_C180s, reductionblocks,
        		    d_SysCx, d_SysCy, d_SysCz,
			    d_sysVCM);
      
        CudaErrorCheck(); 

        CorrectCoMVelocity<<<(No_of_C180s*192)/1024 + 1, 1024>>>(d_velListX, d_velListY, d_velListZ,
                                                                 d_sysVCM,
                                                                 No_of_C180s*192);
          
        CudaErrorCheck(); 
        
        //h_SVCM = d_SVCM;
        //printf("sysVCMx = 	%f, sysVCMy = 		%f, sysVCmz = 		%f\n", h_SVCM[0].x, h_SVCM[0].y, h_SVCM[0].z);
        
 
    }


      if ( step%trajWriteInt == 0 )
      {
          //printf("   Writing trajectory to traj.xyz...\n");
          frameCount++; 
          cudaMemcpy(X, d_X, 192*No_of_C180s*sizeof(float),cudaMemcpyDeviceToHost);
          cudaMemcpy(Y, d_Y, 192*No_of_C180s*sizeof(float),cudaMemcpyDeviceToHost);
          cudaMemcpy(Z, d_Z, 192*No_of_C180s*sizeof(float),cudaMemcpyDeviceToHost);
          cudaMemcpy(CellINdex, d_CellINdex, No_of_C180s*sizeof(int), cudaMemcpyDeviceToHost);
          
          if (binaryOutput)
              WriteBinaryTraj(step + Laststep, trajfile, frameCount + Lastframe);
          else
              write_traj(step + Laststep, trajfile);

          if (write_cont_force == true){

              cudaMemcpy(h_contactForces.x, d_fConList.x, 192*No_of_C180s*sizeof(float), cudaMemcpyDeviceToHost);
              cudaMemcpy(h_contactForces.y, d_fConList.y, 192*No_of_C180s*sizeof(float), cudaMemcpyDeviceToHost);
              cudaMemcpy(h_contactForces.z, d_fConList.z, 192*No_of_C180s*sizeof(float), cudaMemcpyDeviceToHost);
              cudaMemcpy(h_ExtForces.x, d_ExtForces.x, 192*No_of_C180s*sizeof(float), cudaMemcpyDeviceToHost);
              cudaMemcpy(h_ExtForces.y, d_ExtForces.y, 192*No_of_C180s*sizeof(float), cudaMemcpyDeviceToHost);
              cudaMemcpy(h_ExtForces.z, d_ExtForces.z, 192*No_of_C180s*sizeof(float), cudaMemcpyDeviceToHost);
              
              cudaMemcpy(velListX, d_velListX, 192*No_of_C180s*sizeof(float), cudaMemcpyDeviceToHost);
              cudaMemcpy(velListY, d_velListY, 192*No_of_C180s*sizeof(float), cudaMemcpyDeviceToHost);
              cudaMemcpy(velListZ, d_velListZ, 192*No_of_C180s*sizeof(float), cudaMemcpyDeviceToHost);
              
              cudaMemcpy(X, d_X, 192*No_of_C180s, cudaMemcpyDeviceToHost);
              cudaMemcpy(Y, d_Y, 192*No_of_C180s, cudaMemcpyDeviceToHost);
              cudaMemcpy(Z, d_Z, 192*No_of_C180s, cudaMemcpyDeviceToHost);
              
              cudaMemcpy(pressList, d_pressList, No_of_C180s*sizeof(float), cudaMemcpyDeviceToHost);
              cudaMemcpy(volume, d_volume, No_of_C180s*sizeof(float), cudaMemcpyDeviceToHost);
              cudaMemcpy(area, d_area, No_of_C180s*sizeof(float), cudaMemcpyDeviceToHost);	
                    
              writeForces(forceFile, step + Laststep, No_of_C180s);
          }
          if(write_vel_file){
                         
               cudaMemcpy(velListX, d_velListX, 192*No_of_C180s*sizeof(float), cudaMemcpyDeviceToHost);
               cudaMemcpy(velListY, d_velListY, 192*No_of_C180s*sizeof(float), cudaMemcpyDeviceToHost);
               cudaMemcpy(velListZ, d_velListZ, 192*No_of_C180s*sizeof(float), cudaMemcpyDeviceToHost);
          	write_vel(step + Laststep, velFile,frameCount + Lastframe);
          }  
      }

      myError = cudaGetLastError();
      if ( cudaSuccess != myError )
      {
          printf( "Error %d: %s!\n",myError,cudaGetErrorString(myError) );return(-1);
      }
  }


  t = (Time_steps+equiStepCount+Laststep+1) / trajWriteInt; 
  
  if(write_vel_file){
       
       fclose(velFile);   
        
       velFile = fopen("velocity.xyz", "r+");
       fseek(velFile, 0, SEEK_SET);
       fwrite(&No_of_C180s, sizeof(int), 1, velFile);  
       fseek(velFile, 8, SEEK_SET);
       fwrite(&t, sizeof(int), 1, velFile);
       
  }
  
    if (binaryOutput){
  
      fclose(trajfile);
      
      trajfile = fopen (trajFileName, "r+");
      fseek(trajfile, 0, SEEK_SET);
      fwrite(&No_of_C180s, sizeof(int), 1, trajfile);    
      fseek(trajfile, 8, SEEK_SET);
      fwrite(&t, sizeof(int), 1, trajfile);
  }
  
  printf("Xdiv = %d, Ydiv = %d, Zdiv = %d\n", Xdiv, Ydiv, Zdiv );

  FILE* MitIndFile;
  std::fstream MitIndFile2;
  std::string datFileName = inpFile; 
  
  if (overWriteMitInd == 0){
      
      MitIndFile = fopen(mitIndFileName, "a");
      //MitIndFile2.open(datFileName, "a"); 
  }
  else{
      MitIndFile = fopen(mitIndFileName, "w");
      //MitIndFile2.open(datFileName, "w"); 
  }
  if (MitIndFile == NULL)
  {
      printf("Failed to open mit-index.dat\n");
      exit(1);
  }


  for (int i = 0; i < (Time_steps/newCellCountInt) + 1; i++)
  {
      if ( dividingCells[i]!=0 && totalCells[i]!=0 ){
          fprintf(MitIndFile, "%f\n", (float)dividingCells[i]/totalCells[i]);
          // totalCells is number of non-dividing cells
          
      }
      else {
          fprintf(MitIndFile, "%f\n", 0.0);

      }

  }
  
  if ( writeRestartFile(step + Laststep, frameCount + Lastframe) != 0 ){
 	printf("Unable to call Restart Kernel. \n");
	return(-1);
   }
 
 
 
  cudaFree( (void *)d_X  );
  cudaFree( (void *)d_Y  );
  cudaFree( (void *)d_Z  );
  cudaFree( (void *)d_CMx );
  cudaFree( (void *)d_CMy );
  cudaFree( (void *)d_CMz );
  cudaFree( (void *)d_ran2 );

  cudaFree( (void *)d_C180_nn);
  cudaFree( (void *)d_C180_sign);
  cudaFree( (void *)d_cell_div);
  cudaFree( (void *)d_cell_Apo);
  
  cudaFree( (void *)d_DivPlane.x);
  cudaFree( (void *)d_DivPlane.y);
  cudaFree( (void *)d_DivPlane.z);
  
  
  free(X); free(Y); free(Z);
  free(CMx); free(CMy); free(CMz);
  free(dividingCells); free(totalCells);
  free(NoofNNlist);
  free(NNlist);
  free(ran2);
  free(num_new_cells_per_step);
  free(cell_div_inds);
  free(cell_Apo_inds);
  free(pressList);

  free(velListX); 
  free(velListY); 
  free(velListZ); 
  
  free(DivPlane.x);
  free(DivPlane.y);
  free(DivPlane.z);

  fclose(trajfile);
  fclose(forceFile);
  fclose(velFile);
  fclose(MitIndFile);
#ifdef OUTPUT_ADP_ERROR
  fclose(timeFile);
  fclose(errFile);
#endif
  return(0);
}



int initialize_C180s(int Orig_No_of_C180s)
{
  int rank;
  int atom;
  float initx[181], inity[181], initz[181];
  FILE *infil;

  printf("      Initializing positions for %d fullerenes...\n", Orig_No_of_C180s);


  CPUMemory += 3L*192L*MaxNoofC180s*sizeof(float);
  CPUMemory += MaxNoofC180s*6L*sizeof(float);

  infil = fopen("C180","r");
  if ( infil == NULL ) {printf("Unable to open file C180\n");return(-1);}
  for ( atom = 0 ; atom < 180 ; ++atom)
  {
      if ( fscanf(infil,"%f %f %f",&initx[atom], &inity[atom], &initz[atom]) != 3 )
      {
          printf("   Unable to read file C180 on line %d\n",atom+1);
          fclose(infil);
          return(-1);
      }
  }
  fclose(infil);

  // first correct for the cells com

  float sumx = 0; 
  float sumy = 0; 
  float sumz = 0;
      
  for (int i =0; i < 180; ++i){
      sumx += initx[i]; 
      sumy += inity[i]; 
      sumz += initz[i]; 
  }

  sumx /= 180.0; 
  sumy /= 180.0; 
  sumz /= 180.0;

  // calculate initial cell volume

  
      
  for (int i =0; i < 180; ++i){
      initx[i] -= sumx; 
      inity[i] -= sumy; 
      initz[i] -= sumz; 
  }

  float rCheck = powf(0.75*(1.f/3.14159)*0.786, 1.f/3.f); // this code is magical
  printf("Check radius = %f\n", rCheck);
  float3 allCMs[Orig_No_of_C180s];

  float vol = 0;
  int k = 0;
      
  vol = (boxMax.x - BoxMin.x)*(boxMax.y - BoxMin.y)*(boxMax.z - BoxMin.z);
  k = floor(vol/0.786);
      
  if (k < Orig_No_of_C180s){
      fprintf(stderr, "ERROR: Simulation box is too small\n");
      fprintf(stderr, "       Big enough for %d\n", k);
      return 27;
  }

  printf("Can fit upto %d cells\n", k);

  int c = 0;
  float rands[3];
  float3 center = 0.5*boxMax;	
  float3 CM;
  float yoffset;
  yoffset = BoxMin.y + 1;
  if (LineCenter == 1) {
	yoffset = center.y; 
  }

  if (colloidal_dynamics){
  
  	
  	ShapeScaler (initx,inity,initz);
  
  	while (true){
  	
              ranmar(rands, 3);
              CM = make_float3(rands[0]*((boxMax.x - BoxMin.x) - 1.f)  + BoxMin.x + 1.f,
                                      rands[1]*((boxMax.y - BoxMin.y) - 1.f)  + BoxMin.y + 1.f,
                                      0.f);
              if (flatbox == 1){
                  CM.z = (boxMax.z - BoxMin.z)/2;
              } else {
                  CM.z = rands[2]*((boxMax.z - BoxMin.z) - 1.f)  + BoxMin.z + 1.f;
              }

	      	
              bool farEnough = true;
              
              
              farEnough = !(CM.x + ScaleFactor[c]*rCheck*shapeLim > boxMax.x || CM.x-ScaleFactor[c]*rCheck*shapeLim < BoxMin.x ||
                            CM.y+ScaleFactor[c]*rCheck*shapeLim > boxMax.y || CM.y-ScaleFactor[c]*rCheck*shapeLim < BoxMin.y ||
                            CM.z+ScaleFactor[c]*rCheck*shapeLim > boxMax.z || CM.z-ScaleFactor[c]*rCheck*shapeLim < BoxMin.z);
              
              
              for (int nInd = 0; nInd < c; ++nInd){
                  if (mag(allCMs[nInd] - CM) < 1.8*rCheck*shapeLim){
                      //(ScaleFactor[nInd]+ScaleFactor[c])
                      farEnough = false;
                      break;
                  }
              }
          
 
              if (farEnough){

                  allCMs[c] = CM; 
                  c++;
              }
          
              if (c == Orig_No_of_C180s){
		break;
              }
          }
  
  	  if(RandInitDir){	
  	  
  	  	float axis[3];
	  	float RMat[9];
	  	float theta[1];
	  	float tempS[3];
	  	float tempR[3];
	  	
  	  	for (int cellInd = 0; cellInd < Orig_No_of_C180s; cellInd++){
  	  
  	  		axis[0] = 0; 
          		axis[1] = 1; 
          		axis[2] = 0;	 	  
  	  		GetRandomVector(axis);	  
  	  
  	  		ranmar(theta,1);
  	  		theta[0] = theta[0]*2*3.14159265;
  	  
  	  		RotationMatrix(RMat,axis,theta);
  	     
  	        	for(int nodeInd = 0; nodeInd < 180; ++nodeInd){
                  		
                  		tempS[0] = ScaleFactor[cellInd]*initx[nodeInd];
                  		tempS[1] = ScaleFactor[cellInd]*inity[nodeInd];
                  		tempS[2] = ScaleFactor[cellInd]*initz[nodeInd];
                  		
                  		tempR[0] = RMat[0]*tempS[0] + RMat[1]*tempS[1] + RMat[2]*tempS[2];
                  		tempR[1] = RMat[3]*tempS[0] + RMat[4]*tempS[1] + RMat[5]*tempS[2];
                  		tempR[2] = RMat[6]*tempS[0] + RMat[7]*tempS[1] + RMat[8]*tempS[2];
                  		                  		
                		X[cellInd*192 + nodeInd] = tempR[0] + allCMs[cellInd].x;
                  		Y[cellInd*192 + nodeInd] = tempR[1] + allCMs[cellInd].y;
                  		Z[cellInd*192 + nodeInd] = tempR[2] + allCMs[cellInd].z;
  	     		}
  	   
  	   
  	   	}
  	  
  
  	 } else{
  	 
  	 	for (int cellInd = 0; cellInd < Orig_No_of_C180s; cellInd++){
  	     
  	     		for(int nodeInd = 0; nodeInd < 180; ++nodeInd){
                  
                  		X[cellInd*192 + nodeInd] = ScaleFactor[cellInd]*initx[nodeInd] + allCMs[cellInd].x;
                  		Y[cellInd*192 + nodeInd] = ScaleFactor[cellInd]*inity[nodeInd] + allCMs[cellInd].y;
                  		Z[cellInd*192 + nodeInd] = ScaleFactor[cellInd]*initz[nodeInd] + allCMs[cellInd].z;
  	     
  	     		}
  	   
  	   	}
  
  	}


  } else {

  	if (rand_pos){
          
		while (true){
        	      ranmar(rands, 3);
        	      CM = make_float3(rands[0]*((boxMax.x - BoxMin.x) - 1.f)  + BoxMin.x + 1.f,
                                      rands[1]*((boxMax.y - BoxMin.y) - 1.f)  + BoxMin.y + 1.f,
                                      0.f);
        	      if (flatbox == 1){
        	          CM.z = (boxMax.z - BoxMin.z)/2;
        	      } else {
        	          CM.z = rands[2]*((boxMax.z - BoxMin.z) - 1.f)  + BoxMin.z + 1.f;
        	      }

	      	
        	      bool farEnough = true;
              
              
        	      farEnough = !(CM.x+rCheck > boxMax.x || CM.x-rCheck < BoxMin.x ||
        	                    CM.y+rCheck > boxMax.y || CM.y-rCheck < BoxMin.y ||
        	                    CM.z+rCheck > boxMax.z || CM.z-rCheck < BoxMin.z);
              
              
        	      for (int nInd = 0; nInd < c; ++nInd){
        	          if (mag(allCMs[nInd] - CM) < 2*rCheck){
        	              farEnough = false;
        	              break;
        	          }
        	      }
          
 
        	      if (farEnough){
	
        	          allCMs[c] = CM; 
        	          c++;
        	      }
          
        	      if (c == Orig_No_of_C180s){
			break;
        	      }
        	  }
	
  	} else if ( line ){
	
			c = Orig_No_of_C180s-1;
				
			for ( rank = 0; rank < Orig_No_of_C180s - impurityNum ; rank++ )
        	        {
        	                         
        	   	        CM.x = L*rank + 0.5*L + BoxMin.x;
        	    	      	CM.y = yoffset;
        	    	      	CM.z = center.z;
				allCMs[c] = CM; 
        	    	  	c--;

        	   	}
           	
        	   	if (impurity){
        	   	
        	   		while (true){
        		      
				      ranmar(rands, 3);
        	      		      CM = make_float3(rands[0]*((boxMax.x - BoxMin.x) - 1.f)  + BoxMin.x + 1.f,
        	       	                       rands[1]*((boxMax.y - BoxMin.y) - 1.f)  + BoxMin.y + 1.f,
        	       	                       0.f);
        	      			if (flatbox == 1){
        	       		   CM.z = (boxMax.z - BoxMin.z)/2;
        	      			}else {
        	       	   		CM.z = rands[2]*((boxMax.z - BoxMin.z) - 1.f)  + BoxMin.z + 1.f;
        	      			}

	      		
        			      bool farEnough = true;
        	      
        			      farEnough = !(CM.x+rCheck > boxMax.x || CM.x-rCheck < BoxMin.x ||
        			                    CM.y+rCheck > boxMax.y || CM.y-rCheck < BoxMin.y ||
        			                    CM.z+rCheck > boxMax.z || CM.z-rCheck < BoxMin.z );
              	
        			      for (int nInd = Orig_No_of_C180s-1; nInd > c; --nInd){
        			          if (mag(allCMs[nInd] - CM) < 2*rCheck){
        			              farEnough = false;
        			              break;
        			          }
        			      }
          

				      if (farEnough){
	
        			          allCMs[c] = CM; 
        			          c--;
        			      }
        	  
        			      if (c == -1){
					break;
        			      }
        	       	}
				
			}
  	} else if (plane) {
  
  			rCheck *= 1.2;
  			c = Orig_No_of_C180s-1;
  			float l = 2.5;
			int Side = int (((boxMax.x - BoxMin.x) / l) + 0.1 );
			printf(" Max number of initial cells:  %d\n", Side*Side);
			
			for ( rank = 0; rank < Orig_No_of_C180s - impurityNum ; rank++ )
        	        {
                        
        	               ey=rank%Side;
        			ex=rank/Side;         
        	          	CM.x = l*ex + 0.5*l + BoxMin.x;
        	          	CM.y = l*ey + 0.5*l + BoxMin.x;
        	    	      	CM.z = BoxMin.z + 1 ;
				allCMs[c] = CM; 
        	    	  	c--;
        	   	}
  
  
  
  	} else {			
	
			rCheck *= 1.2;
			c = Orig_No_of_C180s-1;
		
        	 	for ( rank = 0; rank < Orig_No_of_C180s - impurityNum; ++rank )
        	 	{
        	 		 ey=rank%Side_length;
        			 ex=rank/Side_length;
        	          	 CM.x = L1*ex + 0.5*L1 + center.x;
        	          	 CM.y = L1*ey + 0.5*L1 + center.y;
        	          	 CM.z = center.z;
        	          	 allCMs[c] = CM;
        	          	 c--;

        	  	}  
	
			if (impurity){	
			
				while (true){
        		      
			      		ranmar(rands, 3);
        	      	      		CM = make_float3(rands[0]*((boxMax.x - BoxMin.x) - 1.f)  + BoxMin.x + 1.f,
        	                       	       rands[1]*((boxMax.y - BoxMin.y) - 1.f)  + BoxMin.y + 1.f,
        	                       	       0.f);
        	      			if (flatbox == 1){
        	       	   		CM.z = (boxMax.z - BoxMin.z)/2;
        	      			}else {
        	          			CM.z = rands[2]*((boxMax.z - BoxMin.z) - 1.f)  + BoxMin.z + 1.f;
        	      			}

	      		
        		      		bool farEnough = true;
              
        		      		farEnough = !(CM.x+rCheck > boxMax.x || CM.x-rCheck < BoxMin.x ||
        		               	     CM.y+rCheck > boxMax.y || CM.y-rCheck < BoxMin.y ||
        		               	     CM.z+rCheck > boxMax.z || CM.z-rCheck < BoxMin.z );
              
        		      		for (int nInd = Orig_No_of_C180s-1; nInd > c; --nInd){
        		          		if (mag(allCMs[nInd] - CM) < 2*rCheck){
        		              			farEnough = false;
        		              			break;
        		          		}
        		      		}
          

			      		if (farEnough){
	
        		          		allCMs[c] = CM; 
        		          		c--;
        		      		}
        	  
        		 	     if (c == -1){
					break;
        		 	     }
        	       	}
	  	 }
   	}

			 


   	for (int cellInd = 0; cellInd < Orig_No_of_C180s; cellInd++){
       	for(int nodeInd = 0; nodeInd < 180; ++nodeInd){
               	   X[cellInd*192 + nodeInd] = initx[nodeInd] + allCMs[cellInd].x;
               	   Y[cellInd*192 + nodeInd] = inity[nodeInd] + allCMs[cellInd].y;
               	   Z[cellInd*192 + nodeInd] = initz[nodeInd] + allCMs[cellInd].z;
       	}
   	}
   }	
       
      // check all the fucking positions...
      for (int i = 0; i < Orig_No_of_C180s*192; ++i){
          if (X[i] > boxMax.x || X[i] < BoxMin.x ||
              Y[i] > boxMax.y || Y[i] < BoxMin.y ||
              Z[i] > boxMax.z || Z[i] < BoxMin.z ){

              printf("shit is in the fan\n");
              printf("%f %f %f\n", X[i], Y[i], Z[i]);
              //exit(4); 
          }
                               
      }


	for (int cellInd = 0; cellInd < Orig_No_of_C180s; cellInd++)
	{
		CellINdex[cellInd] = cellInd;
	}


  return(0);
}


void RotationMatrix(float* RMat,float* axis,float* theta){

	float C,S;
  	C = cos(theta[0]);
  	S = sin(theta[0]);
  	  
  	RMat[0] = 1.0f + (1.0f - C)*(-axis[2]*axis[2] - axis[1]*axis[1]);
  	RMat[1] = -S*axis[2] + (1.0f - C)*axis[0]*axis[1];
  	RMat[2] =  S*axis[1] + (1.0f - C)*axis[0]*axis[2];
  	  
  	RMat[3] =  S*axis[2] + (1.0f - C)*axis[0]*axis[1];
  	RMat[4] =  1.0f + (1.0f - C)*(-axis[2]*axis[2] - axis[0]*axis[0]);
  	RMat[5] = -S*axis[0] + (1.0f - C)*axis[1]*axis[2];
  	  
  	RMat[6] = -S*axis[1] + (1.0f - C)*axis[0]*axis[2];
  	RMat[7] =  S*axis[0] + (1.0f - C)*axis[1]*axis[2];
  	RMat[8] =  1.0f + (1.0f - C)*(-axis[0]*axis[0] - axis[1]*axis[1]);

}


inline void initialize_Plane(int MaxNoofC180s){



   float v[3], w[3];
     
   if (useDivPlaneBasis){
          
              
              if (divPlaneBasis[1] != 0){
        		
        		v[0] = 0;
        		v[1] = divPlaneBasis[2];
        		v[2] = -1*divPlaneBasis[1];

        		w[0] = divPlaneBasis[1];
        		w[1] = -1*divPlaneBasis[0];
        		w[2] = 0;
    		}else{ // this branch is very unlikely, placed for correctness
        		v[0] = 0;
        		v[1] = 1;
        		v[2] = 0;

        		w[0] = divPlaneBasis[2];
        		w[1] = 0;
       		w[2] = -1*divPlaneBasis[0];
    		}

    		// Orthogonalize
   		float f = (w[0]*v[0] + w[1]*v[1] + w[2]*w[2])/(v[0]*v[0] + v[1]*v[1] + v[2]*v[2]);

    		w[0] = w[0] - f*v[0];
    		w[1] = w[1] - f*v[1];
      		w[2] = w[2] - f*v[2];

    		// normalize
    		f = sqrt(v[0]*v[0] + v[1]*v[1] + v[2]*v[2]);

    		v[0] = v[0]/f;
    		v[1] = v[1]/f;
    		v[2] = v[2]/f;

    		f = sqrt(w[0]*w[0] + w[1]*w[1] + w[2]*w[2]);

    		w[0] = w[0]/f;
    		w[1] = w[1]/f;
    		w[2] = w[2]/f;
    
    }
    
    
    for (int i = 0; i < MaxNoofC180s; i++) {
     
          
          float norm[3];
          
          if (useDivPlaneBasis)
              	
              	GetRandomVectorBasis(norm,v,w);
          
          else
          
              GetRandomVector(norm);

#ifdef TURNOFF_RAN

          norm[0] = 0; 
          norm[1] = 1; 
          norm[2] = 0;
          
#endif
          
          DivPlane.x[i] = norm[0];
          DivPlane.y[i] = norm[1];
          DivPlane.z[i] = norm[2]; 

   }

          cudaMemcpy( d_DivPlane.x, DivPlane.x, MaxNoofC180s*sizeof(float), cudaMemcpyHostToDevice);
          cudaMemcpy( d_DivPlane.y, DivPlane.y, MaxNoofC180s*sizeof(float), cudaMemcpyHostToDevice);
          cudaMemcpy( d_DivPlane.z, DivPlane.z, MaxNoofC180s*sizeof(float), cudaMemcpyHostToDevice);
          CudaErrorCheck();

   if (asymDivision){	
   
	ranmar(asym, MaxNoofC180s); 
	 
   } else {
   	for (int i = 0; i < MaxNoofC180s; i++) asym[i] = 0.5;
   }
   
   cudaMemcpy( d_asym, asym, MaxNoofC180s*sizeof(float), cudaMemcpyHostToDevice);
   CudaErrorCheck();	

}




int initialize_Vel(int Orig_No_of_C180s)
{

int rank;
int atom;
float rands[3];
float3 Vcom;

printf("      Initializing velocities for %d fullerenes...\n", Orig_No_of_C180s);
  
	      
	for ( rank = 0; rank < Orig_No_of_C180s; ++rank )
	{	
		ranmar(rands, 3);
		Vcom = make_float3(rands[0]*6 - 3, rands[1]*6 - 3, rands[2]*6 - 3);
		if (flatbox == 1) Vcom.z = 0;
			
               for ( atom = 0 ; atom < 180 ; ++atom)
		{    
                    velListX[rank*192+atom] = Vcom.x;
                    velListY[rank*192+atom] = Vcom.y;
                    velListZ[rank*192+atom] = Vcom.z;
          	}

        }

  return(0);
  
}

int SecondCell (int Orig_No_of_C180s){


  float sumx = 0; 
  float sumy = 0; 
  float sumz = 0;
	  
	  
	 if (closenessToCenter > 0.f && closenessToCenter < 1.f){
          printf("Only making cells within %f of max radius different\n", closenessToCenter);
          	CenterOfMass<<<No_of_C180s,256>>>(No_of_C180s,
                              	              d_X, d_Y, d_Z,
                                      	      d_CMx, d_CMy, d_CMz);
          
         	 cudaMemcpy(CMx, d_CMx, No_of_C180s*sizeof(float), cudaMemcpyDeviceToHost);
         	 cudaMemcpy(CMy, d_CMy, No_of_C180s*sizeof(float), cudaMemcpyDeviceToHost);
         	 cudaMemcpy(CMz, d_CMz, No_of_C180s*sizeof(float), cudaMemcpyDeviceToHost);

         	 float3 sysCM = make_float3(0.f, 0.f, 0.f);

         	 for(int i =0; i < No_of_C180s; ++i){
         	     sysCM = sysCM + make_float3(CMx[i], CMy[i], CMz[i]);
         	 }

         	 sysCM = sysCM/No_of_C180s; 
          
         	 printf("COM = (%f, %f, %f)\n", sysCM.x, sysCM.y, sysCM.z);

         	 float rMax = 0;
         	 float mags[No_of_C180s];
          
         	 for (int i =0; i < No_of_C180s; ++i){
         	     float3 pos = make_float3(CMx[i], CMy[i], CMz[i]) - sysCM;
         	     mags[i] = mag(pos);
         	     rMax = max(rMax, mags[i]);
         	 }
        	 int c = 0; 
          	 for (int i = 0; i < No_of_C180s; ++i){
              		if (mags[i]/rMax <= closenessToCenter){
              		
              			ScaleFactor[i] = SizeFactor;
                  		youngsModArray[i] = Stiffness2;
                  		Growth_rate[i] = gRate;
                  		DivisionVolume[i] = divisionV;
                  		gamma_env[i] = gEnv;
                  		viscotic_damp[i] = gVis;
                  		CellINdex[i] = - CellINdex[i];
                  		
                  		
                  		for (int j =0; j < 180; ++j){
      					sumx += X[i*192 + j]; 
      					sumy += Y[i*192 + j]; 
      					sumz += Z[i*192 + j]; 
  				}

  				sumx /= 180.0; 
  				sumy /= 180.0; 
  				sumz /= 180.0;


  				for (int j =0; j < 180; ++j){
      					X[i*192 + j] -= sumx; 
      					Y[i*192 + j] -= sumy; 
      					Z[i*192 + j] -= sumz; 
  				}
                  		
		
				for(int j = 0; j < 180; ++j){
       	           		X[i*192 + j] = SizeFactor*X[i*192 + j] + sumx;
       	           		Y[i*192 + j] = SizeFactor*Y[i*192 + j] + sumy;
       	           		Z[i*192 + j] = SizeFactor*Z[i*192 + j] + sumz;
       			}

				sumx = 0;
				sumy = 0;
				sumz = 0;
				
                  		++c;
                  		
              		}
          	}
          	
          	printf("Made %d cells softer\n", c);

	  }else{

		printf("Choosing second cell randomly\n");
        	int c = numberOfCells;
                if (fractionOfCells > 0.f && fractionOfCells < 1.f){
                	c = round(fractionOfCells*(float)(Orig_No_of_C180s - impurityNum));
                	
                }

                if (c > Orig_No_of_C180s - impurityNum){
              		printf("ERROR: Too many different cells requested\n");
              		return 12517;
          	}
	 
          	float rands[1];
	  	int coun;
	  	coun = c;	
	  	while(true){
			
	  		ranmar(rands, 1);
                	int i = round(rands[0]*(No_of_C180s - impurityNum) + impurityNum );

			if ( ScaleFactor[i] == SizeFactor ||  youngsModArray[i] == Stiffness2) continue;                 
		
			ScaleFactor[i] = SizeFactor;
			youngsModArray[i] = Stiffness2;
			Growth_rate[i] = gRate;
			DivisionVolume[i] = divisionV;
			gamma_env[i] = gEnv;
			viscotic_damp[i] = gVis;
			CellINdex[i] = - CellINdex[i];
			
                  	for (int j =0; j < 180; ++j){
      				sumx += X[i*192 + j]; 
      				sumy += Y[i*192 + j]; 
      				sumz += Z[i*192 + j]; 
  			}

  			sumx /= 180.0; 
  			sumy /= 180.0; 
  			sumz /= 180.0;


  			for (int j =0; j < 180; ++j){
      				X[i*192 + j] -= sumx; 
      				Y[i*192 + j] -= sumy; 
      				Z[i*192 + j] -= sumz; 
  			}
                  		
		
			for(int j = 0; j < 180; ++j){
       	           	X[i*192 + j] = SizeFactor*X[i*192 + j] + sumx;
       	           	Y[i*192 + j] = SizeFactor*Y[i*192 + j] + sumy;
       	           	Z[i*192 + j] = SizeFactor*Z[i*192 + j] + sumz;
       		}

			sumx = 0;
			sumy = 0;
			sumz = 0;
			
			coun--;
			
			
			if (coun == 0 ) break;

	      }

	}	      

	return 0;

}


int DispersityFunc(int Orig_No_of_C180s){

	
	if(dispersity && colloidal_dynamics && !Restart){
	
		float rands[1];
		
		for (int rank = 0; rank < Orig_No_of_C180s; ++rank )
		{
			ranmar(rands,1);
			ScaleFactor[rank] = rands[0]*0.35 + 0.65 ;
		}

	}

	return(0);

}


void ShapeScaler (float* initX,float* initY,float* initZ){


	
	float sumx = 0; 
  	float sumy = 0; 
  	float sumz = 0;

  	
  	for (int i =0; i < 180; ++i){
      		sumx += initX[i]; 
      		sumy += initY[i]; 
      		sumz += initZ[i]; 
  	}

  	sumx /= 180.0; 
  	sumy /= 180.0; 
  	sumz /= 180.0;  
      
  	for (int i =0; i < 180; ++i){
      		initX[i] -= sumx; 
      		initY[i] -= sumy; 
      		initZ[i] -= sumz; 
  	}
  	
  	
  	for (int i =0; i < 180; ++i){
  	
  		initX[i] *= Xratio;
  		initY[i] *= Yratio;
  		initZ[i] *= Zratio;
  			
  	}
	

}


int generate_random(int no_of_ran1_vectors)
{
  // This function uses marsaglia random number generator
  // Defined in marsaglia.h
  int seed_ij, seed_kl ,ij,kl;

  

  ran2 = (float *)calloc(MaxNoofC180s+1,sizeof(float));
  CPUMemory += (MaxNoofC180s+1L)*sizeof(float);
  if (rand_seed < 0){
      time_t current_time;
      time(&current_time);
      seed_ij = (int)current_time;
      localtime(&current_time);
      seed_kl = (int)current_time;
  
      ij = seed_ij%31328;
      kl = seed_kl%30081;
  } else {
      ij = rand_seed;
      kl = rand_seed*100;
  }

#ifdef TURNOFF_RAN
  ij = 80;
  kl = 97;
#endif
  
  rmarin(ij,kl);

  printf("RNG seeds: %d, %d\n", ij, kl); 
  return(0);
}



int read_fullerene_nn(void)
{
  int i,end;
  int N1, N2, N3, N4, N5, N6, Sign;
  FILE *infil;

  printf("   Reading C180NN ..\n");

  infil = fopen("C180NN","r");
  if ( infil == NULL ) {printf("Unable to open file C180NN\n");return(-1);}
  
  end = 180;
  for ( i = 0; i < 180 ; ++i )
  {
      if ( fscanf(infil,"%d,%d,%d,%d", &N1, &N2, &N3, &Sign) != 4 ) {end = i; break;}
      C180_nn[0 + i] = N1-1;
      C180_nn[192+i] = N2-1;
      C180_nn[384+i] = N3-1;
      C180_sign[i] = Sign;
  }
  fclose(infil);

  if ( end < 180 ) {printf("Error: Unable to read line %d in file C180NN\n",end);return(-1);}

  printf("   Reading C180C ..\n");

  infil = fopen("C180C","r");
  if ( infil == NULL ) {printf("Unable to open file C180C\n");return(-1);}

  end = 270;
  for ( i = 0; i < 270 ; ++i )
  {
      if ( fscanf(infil,"%d,%d", &N1, &N2) != 2 ) {end = i; break;}
      CCI[0][i] = N1-1;
      CCI[1][i] = N2-1;
  }
  fclose(infil);

  if ( end < 270 ) {printf("Error: Unable to read line %d in file C180C\n",end);return(-1);}

  printf("      read nearest neighbour ids for atoms in C180\n");

  printf("   Reading C180 pentagons, hexagons ..\n");

  infil = fopen("C180_pentahexa","r");
  if ( infil == NULL ) {printf("Unable to open file C180_pentahexa\n");return(-1);}

  end = 12;
  for ( i = 0; i < 12 ; ++i )
  {
      if ( fscanf(infil,"%d %d %d %d %d", &N1, &N2, &N3, &N4, &N5) != 5 ) {end = i; break;}
      C180_56[i*7+0] = N1;
      C180_56[i*7+1] = N2;
      C180_56[i*7+2] = N3;
      C180_56[i*7+3] = N4;
      C180_56[i*7+4] = N5;
      C180_56[i*7+5] = N1;
      C180_56[i*7+6] = N1;
  }
  if ( end != 12 ) {printf("Error: Unable to read line %d in file C180_pentahexa\n",end);return(-1);}
  end = 80;
  for ( i = 0; i < 80 ; ++i )
  {
      if ( fscanf(infil,"%d %d %d %d %d %d", &N1, &N2, &N3, &N4, &N5, &N6) != 6 ) {end = i; break;}
      C180_56[84+i*7+0] = N1;
      C180_56[84+i*7+1] = N2;
      C180_56[84+i*7+2] = N3;
      C180_56[84+i*7+3] = N4;
      C180_56[84+i*7+4] = N5;
      C180_56[84+i*7+5] = N6;
      C180_56[84+i*7+6] = N1;
  }
  if ( end != 80 ) {printf("Error: Unable to read line %d in file C180_pentahexa\n",end);return(-1);}

  fclose(infil);

  printf("Calculating equilibrium bond lengths\n");

  float initX[181], initY[181], initZ[181];

  infil = fopen("C180","r");
  if ( infil == NULL ) {printf("Unable to open file C180\n");return(-1);}
  for ( int atom = 0 ; atom < 180 ; ++atom)
  {
  	if ( fscanf(infil,"%f %f %f",&initX[atom], &initY[atom], &initZ[atom]) != 3 )
  	{
              printf("   Unable to read file C180 on line %d\n",atom+1);
              fclose(infil);
              return(-1);
          }
  }
  fclose(infil);

  if (colloidal_dynamics) ShapeScaler (initX,initY,initZ);			

  for (int i = 0; i < 180; ++i){
      int N1 = C180_nn[0 + i];
      int N2 = C180_nn[192 + i];
      int N3 = C180_nn[384 + i];

      float3 a, b; 
      a = make_float3(initX[i], initY[i], initZ[i]);

      b = make_float3(initX[N1], initY[N1], initZ[N1]);
      h_R0[0 + i] = mag(a-b);

      b = make_float3(initX[N2], initY[N2], initZ[N2]);
      h_R0[192 + i] = mag(a-b);

      b = make_float3(initX[N3], initY[N3], initZ[N3]);
      h_R0[384 + i] = mag(a-b);
  }
  
  float3 p, ni, nj, nk;
  for (int n = 0; n<180; n++){
  
          p = make_float3(initX[n], initY[n], initZ[n]); 

          ni = make_float3(initX[C180_nn[0*192 + n]], initY[C180_nn[0*192 + n]], 
                           initZ[C180_nn[0*192 + n]]); 
          
          nj = make_float3(initX[C180_nn[1*192 + n]], initY[C180_nn[1*192 + n]], 
                           initZ[C180_nn[1*192 + n]]);
          
          nk = make_float3(initX[C180_nn[2*192 + n]], initY[C180_nn[2*192 + n]],
                           initZ[C180_nn[2*192 + n]]);

          ni = ni-p;
          nj = nj-p;
          nk = nk-p; 

          theta0[n].aij = acos(dot(ni, nj)/(mag(ni)*mag(nj)));
          
          theta0[n].ajk = acos(dot(nj, nk)/(mag(nj)*mag(nk)));
          
          theta0[n].aik = acos(dot(ni, nk)/(mag(ni)*mag(nk)));

  } 
  
  return(0);
}


int read_json_params(const char* inpFile){
    // Function to parse a json input file using the jsoncpp library

    // variable to hold the root of the json input
    Json::Value inpRoot;
    Json::Reader inpReader;

    std::ifstream inpStream(inpFile);
    std::string inpString((std::istreambuf_iterator<char>(inpStream)),
                          std::istreambuf_iterator<char>());

    bool parsingSuccess = inpReader.parse(inpString, inpRoot);
    if (!parsingSuccess){
        printf("Failed to parse %s\n", inpFile);
        // There must be a way to keep from converting from string to char*
        // Maybe by making inpString a char*
        printf("%s", inpReader.getFormattedErrorMessages().c_str());
        return -1;
    }
    else
        printf("%s parsed successfully\n", inpFile);

    // begin detailed parameter extraction

    Json::Value coreParams = inpRoot.get("core", Json::nullValue);

    // load core simulation parameters
    if (coreParams == Json::nullValue){
        printf("ERROR: Cannot load core simulation parameters\nExiting");
        return -1;
    }
    else {
        MaxNoofC180s = coreParams["MaxNoofC180s"].asInt(); 
        mass = coreParams["particle_mass"].asFloat();
        repulsion_range = coreParams["repulsion_range"].asFloat();
        attraction_range = coreParams["attraction_range"].asFloat();
        repulsion_strength = coreParams["repulsion_strength"].asFloat();
        attraction_strength = coreParams["attraction_strength"].asFloat();
        Youngs_mod = coreParams["Youngs_mod"].asFloat(); 
        stiffness1 = coreParams["stiffFactor1"].asFloat()*Youngs_mod;
        viscotic_damping = coreParams["viscotic_damping"].asFloat();
        internal_damping = coreParams["internal_damping"].asFloat();
        divVol = coreParams["division_Vol"].asFloat();
        Time_steps = coreParams["div_time_steps"].asFloat();
        delta_t = coreParams["time_interval"].asFloat();
        Restart = coreParams["Restart"].asInt();
        trajWriteInt = coreParams["trajWriteInt"].asInt();
        equiStepCount = coreParams["non_div_time_steps"].asInt();

        std::strcpy (trajFileName, coreParams["trajFileName"].asString().c_str());
        binaryOutput = coreParams["binaryOutput"].asBool(); 

        maxPressure = coreParams["maxPressure"].asFloat();
        minPressure = coreParams["minPressure"].asFloat();
        gamma_visc = coreParams["gamma_visc"].asFloat();
        shear_rate = coreParams["shear_rate"].asFloat();
        rMax = coreParams["growth_rate"].asFloat();
        checkSphericity = coreParams["checkSphericity"].asBool();
        constrainAngles = coreParams["constrainAngles"].asBool();
        dt_max = coreParams["dt_max"].asFloat();
        dt_tol = coreParams["dt_tol"].asFloat();
        doAdaptive_dt = coreParams["doAdaptive_dt"].asBool();
        write_cont_force = coreParams["write_cont_force"].asBool();
        write_vel_file = coreParams["write_vel_file"].asBool();
        std::strcpy(forces_file, coreParams["forces_file"].asString().c_str());
        correct_com = coreParams["correct_com"].asBool();
        correct_Vcom = coreParams["correct_Vcom"].asBool();
                                 
    }

    Json::Value countParams = inpRoot.get("counting", Json::nullValue);
    if (countParams == Json::nullValue){
        // countCells = FALSE;
        printf("ERROR: Cannot load counting parameters\nExiting");
        return -1;
    }
    else {
        countCells = countParams["countcells"].asBool();
        std::strcpy(mitIndFileName, countParams["mit-index_file_name"].asString().c_str()); 
        countOnlyInternal = countParams["count_only_internal_cells?"].asBool();
        radFrac = countParams["radius_cutoff"].asFloat();
        overWriteMitInd = countParams["overwrite_mit_ind_file?"].asBool();
        newCellCountInt = countParams["cell_count_int"].asInt();
    }
    

    Json::Value popParams = inpRoot.get("population", Json::nullValue);
    if (popParams == Json::nullValue){
        printf("ERROR: Cannot load population parameters\nExiting");
        return -1;
    }
    else{
        doPopModel = popParams["doPopModel"].asInt();
        totalFood = popParams["totalFood"].asFloat();
        cellFoodCons = popParams["regular_consumption"].asFloat();
        cellFoodConsDiv = popParams["division_consumption"].asFloat();
        cellFoodRel = popParams["death_release_food"].asFloat();
        cellLifeTime = popParams["cellLifeTime"].asInt();
        maxPop = popParams["max_pop"].asFloat(); 
    }

    
    Json::Value apoParams = inpRoot.get("apoptosis", Json::nullValue);
    if (apoParams == Json::nullValue){
        printf("ERROR: Cannot load apoptosis parameters\nExiting");
        return -1;
    }
    else{
	apoptosis = apoParams["apoptosis"].asBool();
    	popToStartApo = apoParams["popToStartApo"].asFloat();
    	Apo_rate = apoParams["Apo_ratio"].asFloat();
    	squeeze_rate = apoParams["squeeze_rate"].asFloat();
    	ApoVol = apoParams["apoptosis_Vol"].asFloat();
    	
    }	

    Json::Value divParams = inpRoot.get("divParams", Json::nullValue);
    
    if (divParams == Json::nullValue){
        printf("ERROR: Cannot load division parameters\n");
        return -1;
    } else{
        useDivPlaneBasis = divParams["useDivPlaneBasis"].asInt();
        divPlaneBasis[0] = divParams["divPlaneBasisX"].asFloat();
        divPlaneBasis[1] = divParams["divPlaneBasisY"].asFloat();
        divPlaneBasis[2] = divParams["divPlaneBasisZ"].asFloat();
	asymDivision = divParams["asymDivision"].asBool();
    }

    Json::Value NewCell = inpRoot.get("NewCell", Json::nullValue);

    if (NewCell == Json::nullValue){
        printf("ERROR: Cannot load New Cell parameters\n");
        return -1;
    } else {
        useDifferentCell = NewCell["useDifferentCell"].asBool();
        SizeFactor = NewCell["SizeFactor"].asFloat();
        Stiffness2 = NewCell["StiffFactor"].asFloat() * Youngs_mod;
        gRate = NewCell["GrowthRate"].asFloat();
        divisionV = NewCell["divisionV"].asFloat();
        gEnv = NewCell["gamma"].asFloat();
        gVis = NewCell["VisDamping"].asFloat();
        Apo_rate2 = NewCell["Apo_rate2"].asFloat();
        squeeze_rate2 = NewCell["squeeze_rate2"].asFloat();
        numberOfCells = NewCell["numberOfCells"].asInt();
        fractionOfCells = NewCell["fractionOfCells"].asFloat();
        closenessToCenter = NewCell["closenessToCenter"].asFloat();
        chooseRandomCellIndices = NewCell["chooseRandomCellIndices"].asBool(); 
        daughtSame = NewCell["daughtSame"].asBool(); 
        duringGrowth = NewCell["duringGrowth"].asBool();
        recalc_r0 = NewCell["recalc_r0"].asBool(); 
    }
    
    Json::Value ColloidParams = inpRoot.get("ColloidParams", Json::nullValue);

    if (ColloidParams == Json::nullValue){
        printf("ERROR: Cannot load Colloidal parameters\n");
        return -1;
    } else {
    
        colloidal_dynamics = ColloidParams["colloidal_dynamics"].asBool();
    	dispersity = ColloidParams["dispersity"].asBool();
    	rand_vel = ColloidParams["rand_vel"].asBool();
        Xratio = ColloidParams["Xratio"].asFloat();
        Yratio = ColloidParams["Yratio"].asFloat();
        Zratio = ColloidParams["Zratio"].asFloat();
        RandInitDir = ColloidParams["RandInitDir"].asBool();
    
    }
  

    Json::Value boxParams = inpRoot.get("boxParams", Json::nullValue);

    if (boxParams == Json::nullValue){
        printf("ERROR: Cannot load box parameters\n");
        return -1;
    } else{
        useRigidSimulationBox = boxParams["useRigidSimulationBox"].asBool();
        usePBCs = boxParams["usePBCs"].asBool();
        useLEbc = boxParams["useLEbc"].asBool();
        useRigidBoxZ = boxParams["useRigidBoxZ"].asBool();
        useRigidBoxY = boxParams["useRigidBoxY"].asBool();
        threshDist = boxParams["threshDist"].asFloat();
        boxMax.x = boxParams["box_len_x"].asFloat();
        boxMax.y = boxParams["box_len_y"].asFloat(); 
        boxMax.z = boxParams["box_len_z"].asFloat();
        BoxMin.x = boxParams["BoxMin_x"].asFloat();
        BoxMin.y = boxParams["BoxMin_y"].asFloat(); 
        BoxMin.z = boxParams["BoxMin_z"].asFloat();
        flatbox = boxParams["flatbox"].asBool();
        LineCenter = boxParams["LineCenter"].asBool();
        rand_pos = boxParams["rand_pos"].asBool();
	impurity = boxParams["impurity"].asBool();
	impurityNum = boxParams["impurityNum"].asInt();
	line = boxParams["line"].asBool();
	plane = boxParams["plane"].asBool();
	
    }

    if (dt_tol > dt_max || dt_max <= 0 || dt_tol < 0){
        printf("ERROR: Invalid time step parameters\n");
        printf("FATAL ERROR\n");
        exit(-60); 
    }

    Json::Value randParams = inpRoot.get("rand_params", Json::nullValue);
	impurity = boxParams["impurity"].asBool();
	line = boxParams["line"].asBool();
    if (randParams == Json::nullValue){
        printf("ERROR: Cannot load randomness parameters\n");
        return -1;
    }
    else {
        add_rands = randParams["add_rands"].asBool();
        rand_seed = randParams["rand_seed"].asInt();
        rand_dist = randParams["rand_dist"].asInt();
        rand_scale_factor = randParams["rand_scale_factor"].asFloat();
    }


    printf("      mass                = %f\n",mass);
    printf("      repulsion range     = %f\n",repulsion_range);
    printf("      attraction range    = %f\n",attraction_range);
    printf("      repulsion strength  = %f\n",repulsion_strength);
    printf("      attraction strength = %f\n",attraction_strength);
    printf("      Youngs modulus      = %f\n",stiffness1);
    printf("      viscotic damping    = %f\n",viscotic_damping);
    printf("      internal damping    = %f\n",internal_damping);
    printf("      division volume     = %f\n",divVol);
    printf("      Time steps          = %d\n",Time_steps);
    printf("      delta t             = %f\n",delta_t);
    printf("      Restart             = %d\n",Restart);
    printf("      trajWriteInterval   = %d\n",trajWriteInt);
    printf("      countOnlyInternal   = %d\n", countOnlyInternal);
    printf("      radFrac             = %f\n", radFrac);
    printf("      newCellCountInt     = %d\n", newCellCountInt);
    printf("      equiStepCount       = %d\n", equiStepCount);
    printf("      trajFileName        = %s\n", trajFileName);
//    printf("      doPopModel          = %d\n", doPopModel);
//    printf("      totalFood           = %f\n", totalFood);
//    printf("      cellFoodCons        = %f\n", cellFoodCons);
//    printf("      cellFoodConsDiv     = %f\n", cellFoodConsDiv);
//    printf("      cellFoodRel         = %f\n", cellFoodRel);
    printf("      thresDist           = %f\n", threshDist);
    printf("      maxPressure         = %f\n", maxPressure);
    printf("      minPressure         = %f\n", minPressure);
    printf("      growth_rate         = %f\n", rMax);
    printf("      squeeze_rate         = %f\n", squeeze_rate);
    printf("      checkSphericity     = %d\n", checkSphericity);
    printf("      gamma_visc          = %f\n", gamma_visc);
    printf("      useDivPlanebasis    = %d\n", useDivPlaneBasis);
    printf("      divPlaneBasisX      = %f\n", divPlaneBasis[0]);
    printf("      divPlaneBasisY      = %f\n", divPlaneBasis[1]);
    printf("      divPlaneBasisZ      = %f\n", divPlaneBasis[2]);
    printf("      useDifferentCell = %d\n", useDifferentCell);
    printf("      SizeFactor  	=%f\n", SizeFactor);
    printf("      Stiffness2  	=%f\n", Stiffness2);
    printf("      GrowthRate2  	=%f\n", gRate);
    printf("      division volume2  	=%f\n", divisionV);
    printf("      gamma_visc2  	=%f\n", gEnv); 
    printf("      viscotic damping2  	=%f\n", gVis);            
    printf("      numberOfCells       = %d\n", numberOfCells);
    printf("      duringGrowth        = %d\n", duringGrowth);
    printf("      closenesstoCenter   = %f\n", closenessToCenter);
    printf("      fractionOfCells     = %f\n", fractionOfCells);
    printf("      chooseRandomCellIndices = %d\n", chooseRandomCellIndices);
    printf("      daughtSame = 	%d\n", daughtSame);
    printf("      recalc_r0           = %d\n", recalc_r0);
    printf("      useRigidSimulationBox = %d\n", useRigidSimulationBox);
    printf("      usePBCs             = %d\n", usePBCs);
    printf("      box_len_x           = %f\n", boxMax.x);
    printf("      box_len_y           = %f\n", boxMax.y);
    printf("      box_len_z           = %f\n", boxMax.z);
    printf("      BoxMin_x            = %f\n", BoxMin.x);
    printf("      BoxMin_y            = %f\n", BoxMin.y);
    printf("      BoxMin_z            = %f\n", BoxMin.z);
    printf("      flatbox             = %d\n", flatbox); 
    printf("      doAdaptive_dt       = %d\n", doAdaptive_dt); 
    printf("      dt_max              = %f\n", dt_max); 
    printf("      dt_tol              = %f\n", dt_tol);
    printf("      add_rands           = %d\n", add_rands);
    printf("      rand_seed           = %d\n", rand_seed);
    printf("      rand_scale_factor   = %f\n", rand_scale_factor);
    printf("      correct_com         = %d\n", correct_com);
    printf("      correct_Vcom         = %d\n", correct_Vcom);    
    printf("      impurityNum         = %d\n", impurityNum);
    printf("      apoptosis           = %d\n",apoptosis);
    printf("      Apoptosis ratio     = %f\n",Apo_rate);
    printf("      apoptosis volume    = %f\n",ApoVol);
    printf("      squeeze rate        = %f\n",squeeze_rate);
    
    
    

    if (asymDivision){
        printf("asymmetric cell division \n");
    }    

    if ( radFrac < 0.4 || radFrac > 0.8 || radFrac < 0 ){
        printf("radFrac not in [0.4, 0.8] setting to 1.\n");
        countOnlyInternal = 0;
    }

    if (trajWriteInt == 0){
        trajWriteInt = 1;
    }

    if (newCellCountInt == 0){
        newCellCountInt = 1;
    }

    if ( trajWriteInt > Time_steps + equiStepCount){
        printf ("Trajectory write interval is too large\n");
        return -1;
    }

    if ((Time_steps + equiStepCount)%trajWriteInt != 0){
        printf ("Invalid trajectory write interval. Time steps must be divisible by it. \n");
        return -1;
    }

    if (newCellCountInt > Time_steps + 1){
        printf("New cell counting interval is too large. \n");
        return -1;
    }

    if (equiStepCount <= 0){
        equiStepCount = 0;
    }

    if (doPopModel != 1){ // This ensures that Pop modelling is only done if this
        // var is only 1
        doPopModel = 0;
    }

    if (maxPressure < 0){
        printf("Invalid maximum pressure value of %f\n", maxPressure);
        printf("Disabling population modelling...");
        doPopModel = 0;
    }


    if ( (useRigidSimulationBox && usePBCs) || (useRigidSimulationBox && useLEbc) || (useLEbc && usePBCs)){
  
      printf ("More than one boundary condition.... \n");
      return -1;
    }	
    
    if ( line && rand_pos && plane){
  
      printf ("More than one initial condition.... \n");
      return -1;
    }

	


    /*

    // The if statement below is not a very good one
    // think about rewriting.
    if (totalFood < 0.0
    || No_of_threads*100 < totalFood
    || cellFoodCons < 0.0
    || cellFoodCons*No_of_threads*10 < totalFood
    || cellFoodConsDiv < 0.0
    || cellFoodConsDiv*No_of_threads*10 < totalFood
    ){
    doPopModel = 0;
    printf("Food parameters invalid. Skipping population modelling.\n");
    }
    */

    if ( !(closenessToCenter >=0 && closenessToCenter <= 1) ){
        printf("ERROR: closenessToCenter is not in [0, 1]\n");
        printf("ERROR: invalid input parameter\n");
        return -1;
    }


    if (fractionOfCells > 1.0){
        printf("ERROR: Softer cell fraction is > 1\n");
        return -1;
    }
        

    return 0;
}


int read_global_params(void)
{
  int error;
  FILE *infil;

  printf("   Reading inp.dat ..\n");

  infil = fopen("inp.dat","r");
  if ( infil == NULL ) {printf("Error: Unable to open file inp.dat\n");return(-1);}

  error = 0;


  if ( fscanf(infil,"%f",&mass)                != 1 ) {error =  1 ;}
  if ( fscanf(infil,"%f",&repulsion_range)     != 1 ) {error =  2 ;}
  if ( fscanf(infil,"%f",&attraction_range)    != 1 ) {error =  3 ;}
  if ( fscanf(infil,"%f",&repulsion_strength)  != 1 ) {error =  4 ;}
  if ( fscanf(infil,"%f",&attraction_strength) != 1 ) {error =  5 ;}
//  if ( fscanf(infil,"%f",&Youngs_mod)          != 1 ) {error =  6 ;}
  if ( fscanf(infil,"%f",&viscotic_damping)    != 1 ) {error =  7 ;}
  if ( fscanf(infil,"%f",&internal_damping)    != 1 ) {error =  8 ;}
  if ( fscanf(infil,"%f",&divVol)              != 1 ) {error =  9 ;}
  if ( fscanf(infil,"%d",&Time_steps)          != 1 ) {error = 10 ;}
  if ( fscanf(infil,"%f",&delta_t)             != 1 ) {error = 11 ;}
  if ( fscanf(infil,"%d",&Restart)             != 1 ) {error = 12 ;}
  if ( fscanf(infil,"%d",&trajWriteInt)        != 1 ) {error = 13 ;}
  if ( fscanf(infil,"%d",&countOnlyInternal)   != 1 ) {error = 14 ;}
  if ( fscanf(infil,"%f",&radFrac)             != 1 ) {error = 15 ;}
  if ( fscanf(infil,"%d",&overWriteMitInd)     != 1 ) {error = 16 ;}
  if ( fscanf(infil,"%d",&newCellCountInt)     != 1 ) {error = 17 ;}
  if ( fscanf(infil,"%d",&equiStepCount)       != 1 ) {error = 18 ;}
  if ( fscanf(infil,"%s",trajFileName)         != 1 ) {error = 19 ;}
  if ( fscanf(infil,"%d",&doPopModel)          != 1 ) {error = 20 ;}
  if ( fscanf(infil,"%f",&totalFood)           != 1 ) {error = 21 ;}
  if ( fscanf(infil,"%f",&cellFoodCons)        != 1 ) {error = 22 ;}
  if ( fscanf(infil,"%f",&cellFoodConsDiv)     != 1 ) {error = 23 ;}
  if ( fscanf(infil,"%f",&cellFoodRel)         != 1 ) {error = 24 ;}
  if ( fscanf(infil,"%d",&haylimit)            != 1 ) {error = 25 ;}
  if ( fscanf(infil,"%d",&cellLifeTime)        != 1 ) {error = 26 ;}
  if ( fscanf(infil,"%f",&maxPressure)         != 1 ) {error = 27 ;}
  if ( fscanf(infil,"%f",&threshDist)          != 1 ) {error = 33 ;}





  fclose(infil);

  if ( error != 0 ){
      printf("   Error reading line %d from file inp.dat\n",error);
      return(-1);
  }

  if ( radFrac < 0.4 || radFrac > 0.8 || radFrac < 0 ){
      printf("radFrac not in [0.4, 0.8] setting to 1.\n");
      countOnlyInternal = 0;
  }

  if (trajWriteInt == 0){
      trajWriteInt = 1;
  }

  if (newCellCountInt == 0){
      newCellCountInt = 1;
  }

  if ( trajWriteInt > Time_steps){
      printf ("Trajectory write interval is too large\n");
      return -1;
  }
  

  if (Time_steps%trajWriteInt != 0){
      printf ("Invalid trajectory write interval. Time steps must be divisible by it. \n");
      return -1;
  }

  if (newCellCountInt > Time_steps){
      printf("New cell counting interval is too large. \n");
      return -1;
  }

  if (equiStepCount <= 0){
    equiStepCount = 0;
  }

  if (doPopModel != 1){ // This ensures that Pop modelling is only done if this
                        // var is only 1
      doPopModel = 0;
  }

  if (maxPressure < 0){
      printf("Invalid maximum pressure value of %f\n", maxPressure);
      printf("Disabling population modelling...");
      doPopModel = 0;
  }


  /*

  // The if statement below is not a very good one
  // think about rewriting.
  if (totalFood < 0.0
      || No_of_threads*100 < totalFood
      || cellFoodCons < 0.0
      || cellFoodCons*No_of_threads*10 < totalFood
      || cellFoodConsDiv < 0.0
      || cellFoodConsDiv*No_of_threads*10 < totalFood
       ){
      doPopModel = 0;
      printf("Food parameters invalid. Skipping population modelling.\n");
  }
  */


  printf("      mass                = %f\n",mass);
  printf("      repulsion range     = %f\n",repulsion_range);
  printf("      attraction range    = %f\n",attraction_range);
  printf("      repulsion strength  = %f\n",repulsion_strength);
  printf("      attraction strength = %f\n",attraction_strength);
//  printf("      Youngs modulus      = %f\n",Youngs_mod);
  printf("      viscotic damping    = %f\n",viscotic_damping);
  printf("      internal damping    = %f\n",internal_damping);
  printf("      division volume     = %f\n",divVol);
  printf("      Time steps          = %d\n",Time_steps);
  printf("      delta t             = %f\n",delta_t);
  printf("      Restart             = %d\n",Restart);
  printf("      trajWriteInterval   = %d\n",trajWriteInt);
  printf("      countOnlyInternal   = %d\n", countOnlyInternal);
  printf("      radFrac             = %f\n", radFrac);
  printf("      newCellCountInt     = %d\n", newCellCountInt);
  printf("      equiStepCount       = %d\n", equiStepCount);
  printf("      trajFileName        = %s\n", trajFileName);
//  printf("      doPopModel          = %d\n", doPopModel);
//  printf("      totalFood           = %f\n", totalFood);
//  printf("      cellFoodCons        = %f\n", cellFoodCons);
//  printf("      cellFoodConsDiv     = %f\n", cellFoodConsDiv);
//  printf("      cellFoodRel         = %f\n", cellFoodRel);
  printf("      thresDist           = %f\n", threshDist);


  return(0);
}




//C *****************************************************************



void write_traj(int t_step, FILE* trajfile)
{

  fprintf(trajfile, "%d\n", No_of_C180s * 192);
  fprintf(trajfile, "Step: %d frame: %d\n", t_step, t_step/trajWriteInt);
  
  if (useDifferentCell){
      for (int c = 0; c < No_of_C180s; c++){
          if (youngsModArray[c] == stiffness1)
              fprintf(trajfile, "cell: %d H\n", CellINdex[c]);
          else if(youngsModArray[c] == Stiffness2)
              fprintf(trajfile, "cell: %d C\n", CellINdex[c]);
          else
              fprintf(trajfile, "cell: %d UnknownStiffness\n", CellINdex[c]);

          for (int p = 0; p < 192; p++)
          {
              fprintf(trajfile, "%.7f,  %.7f,  %.7f\n", X[(c*192)+p], Y[(c*192)+p], Z[(c*192)+p]);
          }
      }
        
  } else {
      for (int c = 0; c < No_of_C180s; c++){
              fprintf(trajfile, "cell: %d\n", CellINdex[c]);
              
              for (int p = 0; p < 192; p++)
              {
                  fprintf(trajfile, "%.7f,  %.7f,  %.7f\n", X[(c*192)+p], Y[(c*192)+p], Z[(c*192)+p]);
              }
      }
      
  }
}

void WriteBinaryTraj(int t_step, FILE* trajFile, int frameCount){
    
    fwrite(&t_step, sizeof(int), 1, trajFile);
    fwrite(&frameCount, sizeof(int), 1, trajFile); 
    fwrite(&No_of_C180s, sizeof(int), 1, trajFile);
    if (useDifferentCell){
        int cellType = 0; 
        for (int c = 0; c < No_of_C180s; c++){ 
            fwrite(&CellINdex[c], sizeof(int), 1, trajFile);
            fwrite(X + (c*192), sizeof(float), 192, trajFile); 
            fwrite(Y + (c*192), sizeof(float), 192, trajFile); 
            fwrite(Z + (c*192), sizeof(float), 192, trajFile);
            
            if (youngsModArray[c] == stiffness1)
                cellType = 0;
            else
                cellType = 1; 
            
            fwrite(&cellType, sizeof(int), 1, trajFile);
        }
    } else {
        for (int c = 0; c < No_of_C180s; c++){
        
            fwrite(&CellINdex[c], sizeof(int), 1, trajFile);            
            fwrite(X + (c*192), sizeof(float), 192, trajFile); 
            fwrite(Y + (c*192), sizeof(float), 192, trajFile); 
            fwrite(Z + (c*192), sizeof(float), 192, trajFile); 
        }
    }
        
    
}

void write_vel(int t_step, FILE* velFile,int frameCount){
    
    fwrite(&t_step, sizeof(int), 1, velFile);
    fwrite(&frameCount, sizeof(int), 1, velFile); 
    fwrite(&No_of_C180s, sizeof(int), 1, velFile);
    
    if (useDifferentCell){
        int cellType = 0; 
        for (int c = 0; c < No_of_C180s; c++){
            fwrite(&c, sizeof(int), 1, velFile);
            fwrite(velListX + (c*192), sizeof(float), 192, velFile); 
            fwrite(velListY + (c*192), sizeof(float), 192, velFile); 
            fwrite(velListZ + (c*192), sizeof(float), 192, velFile);
            
            if (youngsModArray[c] == stiffness1)
                cellType = 0;
            else
                cellType = 1; 
            
            fwrite(&cellType, sizeof(int), 1, velFile);
        }
    } else {
        for (int c = 0; c < No_of_C180s; c++){
        
            fwrite(&c, sizeof(int), 1, velFile);
            fwrite(velListX + (c*192), sizeof(float), 192, velFile); 
            fwrite(velListY + (c*192), sizeof(float), 192, velFile); 
            fwrite(velListZ + (c*192), sizeof(float), 192, velFile);
        }
    }
}


inline void count_and_get_div(){
  num_cell_div = 0;
  cudaMemcpy(cell_div, d_cell_div, No_of_C180s*sizeof(char), cudaMemcpyDeviceToHost);
  
  for (int cellInd = 0; cellInd < No_of_C180s; cellInd++) {
    if (cell_div[cellInd] == 1){
      cell_div[cellInd] = 0;
      cell_div_inds[num_cell_div] = cellInd;
      num_cell_div++;
    }
  }
  cudaMemcpy(d_cell_div, cell_div, No_of_C180s*sizeof(char), cudaMemcpyHostToDevice);

#ifdef PRINT_VOLUMES
        if (num_cell_div > 0){
            printf("Dividing cells: ");
            for (int i = 0; i<num_cell_div; i++){
                printf("%d ", cell_div_inds[i]);
            }
          
            printf("\n");
        }
#endif

}


inline void count_and_die(){
  
  num_cell_Apo = 0;
  cudaMemcpy(cell_Apo, d_cell_Apo, No_of_C180s*sizeof(char), cudaMemcpyDeviceToHost);
  cudaMemcpy(d_Growth_rate, Growth_rate, No_of_C180s*sizeof(float), cudaMemcpyHostToDevice);
  
  for (int cellInd = 0; cellInd < No_of_C180s; cellInd++) {
    if (cell_Apo[cellInd] == 1){ 
      if(Growth_rate[cellInd] == -squeeze_rate || Growth_rate[cellInd] == -squeeze_rate2){
      	cell_Apo[cellInd] = 0;
      	cell_Apo_inds[num_cell_Apo] = cellInd;
      	num_cell_Apo++;
      }
    }
  }
  cudaMemcpy(d_cell_Apo, cell_Apo, No_of_C180s*sizeof(char), cudaMemcpyHostToDevice);
}


inline float getRmax2(){
  float dx, dy, dz, Rmax2 = 0;
  for (int cell = 0; cell < No_of_C180s; cell++) {
    dx = CMx[cell] - *h_sysCM.x;
    dy = CMy[cell] - *h_sysCM.y;
    dz = CMz[cell] - *h_sysCM.z;

    Rmax2 = max(Rmax2, dx*dx + dy*dy + dz*dz);

  }

  return Rmax2;

}

inline int num_cells_far(){

  if (num_cell_div == 0 || No_of_C180s < 50) return 0;



  float dx, dy, dz, dr2;
  float Rmax2 = getRmax2();
  int farCellCount = 0;


  for (int cell = No_of_C180s - num_cell_div; cell < No_of_C180s; cell++) { // Only check the newest cells
    dx = CMx[cell] - *h_sysCM.x;
    dy = CMy[cell] - *h_sysCM.y;
    dz = CMz[cell] - *h_sysCM.z;

    dr2 = dx*dx + dy*dy + dz*dz;

    if (dr2 > radFrac*radFrac*Rmax2)
      farCellCount++;
  }

  return farCellCount;

}

void writeForces(FILE* forceFile, int t_step, int num_cells){
    if(forceFile == NULL){
        printf("ERROR: forces file not available\n");
        exit(1);
    }

    for (int c =0; c < num_cells; ++c){
        for (int n = 0; n < 180; ++n){
            fprintf(forceFile, "%d,%d,%d,%d,%d,%.8f,%.8f,%.8f,%.8f,%.8f,%.8f,%.8f,%.8f,%.8f,%.8f,%.8f,%.8f,%.8f,%.8f,%.8f,%.8f,%.8f,%.8f\n",
                    t_step, num_cells, c, n, c*192+n,
                    h_contactForces.x[c*192 + n],
                    h_contactForces.y[c*192 + n],
                    h_contactForces.z[c*192 + n],
                    mag(make_float3(h_contactForces.x[c*192 + n],
                                    h_contactForces.y[c*192 + n],
                                    h_contactForces.z[c*192 + n])),
                                        h_ExtForces.x[c*192 + n],
                    h_ExtForces.y[c*192 + n],
                    h_ExtForces.z[c*192 + n],
                    mag(make_float3(h_ExtForces.x[c*192 + n],
                                    h_ExtForces.y[c*192 + n],
                                    h_ExtForces.z[c*192 + n])),
                    velListX[c*192+n],
                    velListY[c*192+n],
                    velListZ[c*192+n],
                    mag(make_float3(velListX[c*192+n],
                                    velListY[c*192+n],
                                    velListZ[c*192+n])),
                    X[c*192+n],
                    Y[c*192+n],
                    Z[c*192+n],
                    pressList[c],
                    volume[c],
                    area[c]
                );
                        
        }


    }
}





int writeRestartFile(int t_step, int frameCount){
	
	FILE *Restartfile;
	Restartfile = fopen ("Restart.xyz", "w");
        if ( Restartfile == NULL)
  	{
      	    printf("Failed to open Restart file \n" );
      	    return -1;
  	}

	float p = 0;
	float y = 0;
	float g = 0;
	float s = 0;
	float d = 0;
	float ge = 0;
	float vd = 0;
	int I = 0;	

	fwrite(&t_step, sizeof(int), 1, Restartfile);
	fwrite(&frameCount, sizeof(int), 1, Restartfile); 
	fwrite(&No_of_C180s, sizeof(int), 1, Restartfile); 
	fwrite(&impurityNum, sizeof(int), 1, Restartfile);
	fwrite(&NumApoCell, sizeof(int), 1, Restartfile);
	fwrite(&NumRemoveCell, sizeof(int), 1, Restartfile);  
 
        	
        for (int c = 0; c < No_of_C180s; c++){
		p = pressList[c];
		y = youngsModArray[c];
        	g = Growth_rate[c];
        	s = ScaleFactor[c];
        	d = DivisionVolume[c];
        	ge = gamma_env[c];
        	vd =  viscotic_damp[c];
        	I = CellINdex[c];
        		
		fwrite(&c, sizeof(int), 1, Restartfile);
        	fwrite(X + (c*192), sizeof(float), 192, Restartfile); 
        	fwrite(Y + (c*192), sizeof(float), 192, Restartfile); 
        	fwrite(Z + (c*192), sizeof(float), 192, Restartfile);
		fwrite(velListX + (c*192), sizeof(float), 192, Restartfile);
            	fwrite(velListY + (c*192), sizeof(float), 192, Restartfile);
		fwrite(velListZ + (c*192), sizeof(float), 192, Restartfile);
		fwrite(&p, sizeof(float), 1, Restartfile);
            	fwrite(&y, sizeof(float), 1, Restartfile);
            	fwrite(&g, sizeof(float), 1, Restartfile);
            	fwrite(&s, sizeof(float), 1, Restartfile);
            	fwrite(&d, sizeof(float), 1, Restartfile);
            	fwrite(&ge, sizeof(float), 1, Restartfile);
            	fwrite(&vd, sizeof(float), 1, Restartfile);
            	fwrite(&I, sizeof(int), 1, Restartfile);
            	
        }

   
   fclose(Restartfile);
   return 0;

}

int ReadRestartFile( ){

  FILE *infil;
  int s;
  int f;
  int nCell;
  int nImp;  
  int CellInd;
  int shift;
  int NCA;
  int NCR;
  

  printf("Reading Restart.xyz ...\n");
  infil = fopen("Restart.xyz","rb");
  
  if ( infil == NULL ) {
    printf("Unable to open file Restart.xyz \n");
    return(-1);
  }



  if ( fread(&s, sizeof(int),1,infil) != 1 ){ 
	printf("Data missing from trajectory. \n");
	return(-1);
  } else printf("\nstep %d \n",s -1);

  if ( fread(&f, sizeof(int),1,infil) != 1 ){ 
	printf("Data missing from trajectory. \n");
	return(-1);
  } else printf("frame number is: %d \n",f - 1);

  if ( fread(&nCell, sizeof(int),1,infil) != 1 ) { 
	printf("Data missing from trajectory. \n");
	return(-1);
  }
  
  if ( fread(&nImp, sizeof(int),1,infil) != 1 ) { 
	printf("Data missing from trajectory. \n");
	return(-1);
  }

  if ( fread(&NCA, sizeof(int),1,infil) != 1 ) { 
	printf("Data missing from trajectory. \n");
	return(-1);
  }

  if ( fread(&NCR, sizeof(int),1,infil) != 1 ) { 
	printf("Data missing from trajectory. \n");
	return(-1);
  }

  Laststep = s-1;
  Lastframe = f-1;
  No_of_threads = nCell;	
  No_of_C180s = nCell;
  Orig_No_of_C180s = nCell;
  impurityNum = nImp;
  NumApoCell = NCA;
  NumRemoveCell = NCR;  

  printf("Number of the initial Cells is: %d \n",Orig_No_of_C180s);
  printf("Number of the  impurity is: %d \n",impurityNum);
  

  for (int c = 0; c < Orig_No_of_C180s; c++){

    shift = c*192;
    
    if ( fread(&CellInd, sizeof(int),1,infil) != 1 ) printf("Data missing from trajectory. \n");
    if ( fread(&X[shift], sizeof(float),192,infil) != 192 ) printf("Data missing from trajectory. \n");
    if ( fread(&Y[shift], sizeof(float),192,infil) != 192 ) printf("Data missing from trajectory. \n");
    if ( fread(&Z[shift], sizeof(float),192,infil) != 192 ) printf("Data missing from trajectory. \n");
    if ( fread(&velListX[shift], sizeof(float),192,infil) != 192 ) printf("Data missing from trajectory. \n");
    if ( fread(&velListY[shift], sizeof(float),192,infil) != 192 ) printf("Data missing from trajectory. \n");
    if ( fread(&velListZ[shift], sizeof(float),192,infil) != 192 ) printf("Data missing from trajectory. \n");
    if ( fread(&pressList[c], sizeof(float),1,infil) != 1 ) printf("Data missing from trajectory. \n");
    if ( fread(&youngsModArray[c], sizeof(float),1,infil) != 1 ) printf("Data missing from trajectory. \n");
    if ( fread(&Growth_rate[c], sizeof(float),1,infil) != 1 ) printf("Data missing from trajectory. \n");
    if ( fread(&ScaleFactor[c], sizeof(float),1,infil) != 1 ) printf("Data missing from trajectory. \n");
    if ( fread(&DivisionVolume[c], sizeof(float),1,infil) != 1 ) printf("Data missing from trajectory. \n");
    if ( fread(&gamma_env[c], sizeof(float),1,infil) != 1 ) printf("Data missing from trajectory. \n");
    if ( fread(&viscotic_damp[c], sizeof(float),1,infil) != 1 ) printf("Data missing from trajectory. \n");
    if ( fread(&CellINdex[c], sizeof(int),1,infil) != 1 ) printf("Data missing from trajectory. \n");
	
   }

   fclose(infil);
   return 0;

}


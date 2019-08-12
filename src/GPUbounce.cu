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

float mass;                                        //  M
float repulsion_range,    attraction_range;        //  LL1, LL2
float repulsion_strength, attraction_strength;     //  ST1, ST2

// variables to allow for different stiffnesses
float stiffness1;
float stiffness2;
float Youngs_mod; 
float* d_Youngs_mod;
float* youngsModArray; 
bool useDifferentStiffnesses;
bool recalc_r0; 
int* d_indSoftCells;
float softYoungsMod;
int numberOfSofterCells;
bool duringGrowth;
bool daughtSameStiffness;
float closenessToCenter;
int startAtPop;
bool checkSphericity; 

bool chooseRandomCellIndices;
float fractionOfSofterCells;
int phase_count = INT_MAX;
bool write_cont_force=false;
char forces_file[256];


float viscotic_damping, internal_damping;          //  C, DMP
float gamma_visc;
float zOffset; // Offset from Z = 0 for starting positions.
int ranZOffset;
int   Time_steps;
float divVol;
float delta_t;
float dt_max;
float dt_tol;
bool doAdaptive_dt;
float c1 = 0; float c2 = 0; 
int   Restart;
int   trajWriteInt; // trajectory write interval
int   countOnlyInternal; // 0 - Count all new cells
                         // 1 - Count only the cells born within 0.6Rmax from
                         //     the center of mass of the system
float radFrac; // The factor to count cells within a raduys (<Rmax)

int   overWriteMitInd; // 0 No, 1 yes

int newCellCountInt; // Interval at which to count the divided cells
int equiStepCount;
const char* ptrajFileName;
char trajFileName[256];
bool binaryOutput; 

// equilibrium length of springs between fullerene atoms
float R0  = 0.13517879937327418f; // why is this hard-coded?
float* d_R0;
float* h_R0;


float L1  = 3.0f;       // the initial fullerenes are placed in
// an X x Y grid of sizne L1 x L1


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

float *d_volume;
float *volume;
float *d_area; 
float *area; 
char* cell_div;
char* d_cell_div;
int num_cell_div;
int* cell_div_inds;

char mitIndFileName[256]; 

float *d_pressList;
float *pressList;
int* d_resetIndices;
int* resetIndices; 


float* d_velListX; 
float* d_velListY; 
float* d_velListZ;

float* velListX; 
float* velListY; 
float* velListZ; 

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
float maxPressure;
float minPressure;
float rMax;
float maxPop; 

// Params related to having walls in the simulation
int useWalls;
char perpAxis[2];
float threshDist;
float dAxis;
float wallLen;
float wallWidth;
float wall1, wall2;
float wallWStart, wallWEnd;
float wallLStart, wallLEnd;

float boxLength, boxMin[3];
float3 boxMax;
bool flatbox; 
bool useRigidSimulationBox;
bool usePBCs; 
float* d_boxMin;
bool rand_pos; 

// randomness parameters

bool add_rands;
int rand_seed;
int rand_dist;
float rand_scale_factor;
curandState *d_rngStates;
unsigned int *d_seeds; 

int No_of_threads; // ie number of staring cells
int Side_length;
int ex, ey;


float  *X,  *Y,  *Z;     // host: atom positions

float *d_XP, *d_YP, *d_ZP;     // device: time propagated atom positions
float  *d_X,  *d_Y,  *d_Z;     // device: present atom positions
float *d_XM, *d_YM, *d_ZM;     // device: previous atom positions
float *d_XMM, *d_YMM, *d_ZMM; 

R3Nptrs d_fConList;
R3Nptrs d_fDisList;
R3Nptrs d_fRanList; 

R3Nptrs d_contactForces;
R3Nptrs h_contactForces;

float* d_Fx;
float* d_Fy;
float* d_Fz;

// float* theta0;
// float* d_theta0;

bool constrainAngles;

// host: minimal bounding box for fullerene
float *bounding_xyz;
float *d_bounding_xyz;   // device:  bounding_xyz

// global minimum and maximum of x and y, preprocessfirst
// global minimum and maximum of x and y, postprocesssecond
float *d_Minx, *d_Maxx, *d_Miny, *d_Maxy, *d_Minz, *d_Maxz;
float *Minx, *Maxx, *Miny, *Maxy, *Minz, *Maxz;

float DL;
int Xdiv, Ydiv, Zdiv;

//int *d_NoofNNlist;
int *d_NNlist;
int *NoofNNlist;
int *NNlist;

float *d_CMx, *d_CMy, *d_CMz;
float *CMx, *CMy, *CMz;
float sysCMx = 1.0, sysCMy = 1.0, sysCMz = 1.0;
float sysCMx_old = 0.0, sysCMy_old = 0.0, sysCMz_old = 0.0;

//float Pressure;          // pressure
//float Temperature;       // equation of state relates Pressure and Temperature

int  No_of_C180s;        // the global number of C180 fullerenes
int  No_of_C180s_in;     // the number of C180s near the center of mass of the system
int MaxNoofC180s; 

float *ran2;             // host: ran2[]
float *d_ran2;           // device: ran2[], used in celldivision

int *NDIV;               // # of divisions

// Parameters related to division
bool useDivPlaneBasis;
float divPlaneBasis[3]; 

long int GPUMemory;
long int CPUMemory;


int frameCount = 1;

bool correct_com = false; 

int main(int argc, char *argv[])
{
  int i;
  int globalrank,step;
  int noofblocks, threadsperblock, prevnoofblocks;
  int Orig_No_of_C180s, newcells;
  int reductionblocks;
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

  Side_length   = (int)( sqrt( (double)No_of_threads )+0.5);
  if ( No_of_threads > MaxNoofC180s // Side_length*Side_length != No_of_threads
      )
  {
      printf("Usage: Celldiv no_of_threads\n");
      printf("       no_of_threads should be a square, n^2, < %d\n", MaxNoofC180s);
      return(0);
  }


  No_of_C180s      = No_of_threads;
  Orig_No_of_C180s = No_of_C180s;
  GPUMemory = 0L;
  CPUMemory = 0L;

  h_R0 = (float *)calloc(192*3, sizeof(float));

  //if ( read_global_params()               != 0 ) return(-1);
  
  if ( generate_random(Orig_No_of_C180s)  != 0 ) return(-1);
  if ( initialize_C180s(Orig_No_of_C180s) != 0 ) return(-1);
  if ( read_fullerene_nn()                != 0 ) return(-1);
  NDIV = (int *)calloc(MaxNoofC180s,sizeof(int));
  CPUMemory += MaxNoofC180s*sizeof(int);
  for ( i = 0; i < No_of_threads; ++i ) NDIV[i] = 1;
  for ( i = No_of_threads; i < MaxNoofC180s; ++i ) NDIV[i] = 0;

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

  cudaDeviceProp deviceProp = getDevice();
  if (cudaSuccess != cudaSetDevice(atoi(argv[3]))){
      CudaErrorCheck();
      printf("Could not set to divice %d\n", atoi(argv[3]));
      return -1;
  }

  if ( cudaSuccess != cudaMalloc( (void **)&d_C180_nn, 3*192*sizeof(int))) return(-1);
  if ( cudaSuccess != cudaMalloc( (void **)&d_C180_sign, 180*sizeof(int))) return(-1);
  GPUMemory +=  3*192*sizeof(int) + 180*sizeof(int);
  //  cudaError_t myError = cudaGetLastError();
  //     if ( cudaSuccess != myError )
  //         { printf( "1: Error %d: %s!\n",myError,cudaGetErrorString(myError) );return(-1);}

  if ( cudaSuccess != cudaMalloc( (void **)&d_XP , 192*MaxNoofC180s*sizeof(float))) return(-1);
  if ( cudaSuccess != cudaMalloc( (void **)&d_YP , 192*MaxNoofC180s*sizeof(float))) return(-1);
  if ( cudaSuccess != cudaMalloc( (void **)&d_ZP , 192*MaxNoofC180s*sizeof(float))) return(-1);
  if ( cudaSuccess != cudaMalloc( (void **)&d_X  , 192*MaxNoofC180s*sizeof(float))) return(-1);
  if ( cudaSuccess != cudaMalloc( (void **)&d_Y  , 192*MaxNoofC180s*sizeof(float))) return(-1);
  if ( cudaSuccess != cudaMalloc( (void **)&d_Z  , 192*MaxNoofC180s*sizeof(float))) return(-1);
  if ( cudaSuccess != cudaMalloc( (void **)&d_XM , 192*MaxNoofC180s*sizeof(float))) return(-1);
  if ( cudaSuccess != cudaMalloc( (void **)&d_YM , 192*MaxNoofC180s*sizeof(float))) return(-1);
  if ( cudaSuccess != cudaMalloc( (void **)&d_ZM , 192*MaxNoofC180s*sizeof(float))) return(-1);
  if ( cudaSuccess != cudaMalloc( (void **)&d_bounding_xyz , MaxNoofC180s*6*sizeof(float))) return(-1);
  if ( cudaSuccess != cudaMalloc( (void **)&d_CMx ,          MaxNoofC180s*sizeof(float))) return(-1);
  if ( cudaSuccess != cudaMalloc( (void **)&d_CMy ,          MaxNoofC180s*sizeof(float))) return(-1);
  if ( cudaSuccess != cudaMalloc( (void **)&d_CMz ,          MaxNoofC180s*sizeof(float))) return(-1);
//  if ( cudaSuccess != cudaMalloc( (void **)&d_volume ,       MaxNoofC180s*sizeof(float))) return(-1);
  if ( cudaSuccess != cudaMalloc( (void **)&d_area ,       MaxNoofC180s*sizeof(float))) return(-1);
  if ( cudaSuccess != cudaMalloc( (void **)&d_cell_div ,     MaxNoofC180s*sizeof(char))) return(-1);
  if ( cudaSuccess != cudaMalloc( (void **)&d_Minx ,         1024*sizeof(float))) return(-1);
  if ( cudaSuccess != cudaMalloc( (void **)&d_Maxx ,         1024*sizeof(float))) return(-1);
  if ( cudaSuccess != cudaMalloc( (void **)&d_Miny ,         1024*sizeof(float))) return(-1);
  if ( cudaSuccess != cudaMalloc( (void **)&d_Maxy ,         1024*sizeof(float))) return(-1);
  if ( cudaSuccess != cudaMalloc( (void **)&d_Minz ,         1024*sizeof(float))) return(-1);
  if ( cudaSuccess != cudaMalloc( (void **)&d_Maxz ,         1024*sizeof(float))) return(-1);
  //  if ( cudaSuccess != cudaMalloc( (void **)&d_NoofNNlist ,   1024*1024*sizeof(int))) return(-1);
  if ( cudaSuccess != cudaMalloc( (void **)&d_NNlist ,    MAX_NN*MaxNoofC180s*sizeof(int))) return(-1); if ( cudaSuccess != cudaMalloc( (void **)&d_C180_56,       92*7*sizeof(int))) return(-1);
  if ( cudaSuccess != cudaMalloc( (void **)&d_ran2 , 10000*sizeof(float))) return(-1);
  if ( cudaSuccess != cudaMalloc( (void **)&d_pressList, MaxNoofC180s*sizeof(float))) return(-1);
  // if ( cudaSuccess != cudaMalloc( (void **)&d_velListX, 192*MaxNoofC180s*sizeof(float))) return(-1);
  // if ( cudaSuccess != cudaMalloc( (void **)&d_velListY, 192*MaxNoofC180s*sizeof(float))) return(-1);
  // if ( cudaSuccess != cudaMalloc( (void **)&d_velListZ, 192*MaxNoofC180s*sizeof(float))) return(-1);
  if ( cudaSuccess != cudaMalloc( (void **)&d_resetIndices, MaxNoofC180s*sizeof(int))) return(-1);
  if ( cudaSuccess != cudaMalloc( (void **)&d_Fx, 192*MaxNoofC180s*sizeof(float))) return(-1);
  if ( cudaSuccess != cudaMalloc( (void **)&d_Fy, 192*MaxNoofC180s*sizeof(float))) return(-1);
  if ( cudaSuccess != cudaMalloc( (void **)&d_Fz, 192*MaxNoofC180s*sizeof(float))) return(-1);
  if ( cudaSuccess != cudaMalloc( (void **)&d_Youngs_mod, MaxNoofC180s*sizeof(float))) return(-1);
  if ( cudaSuccess != cudaMalloc( (void **)&d_boxMin, 3*sizeof(float))) return(-1);
  if ( cudaSuccess != cudaMalloc( (void **)&d_R0, 192*3*sizeof(float))) return(-1);
  if ( cudaSuccess != cudaMalloc((void **)&d_velListX, 192*MaxNoofC180s*sizeof(float))) return -1; 
  if ( cudaSuccess != cudaMalloc((void **)&d_velListY, 192*MaxNoofC180s*sizeof(float))) return -1; 
  if ( cudaSuccess != cudaMalloc((void **)&d_velListZ, 192*MaxNoofC180s*sizeof(float))) return -1;
  if ( cudaSuccess != cudaMalloc((void **)&d_contactForces.x, 192*MaxNoofC180s*sizeof(float))) return -1;
  if ( cudaSuccess != cudaMalloc((void **)&d_contactForces.y, 192*MaxNoofC180s*sizeof(float))) return -1;
  if ( cudaSuccess != cudaMalloc((void **)&d_contactForces.z, 192*MaxNoofC180s*sizeof(float))) return -1;
  if ( cudaSuccess != cudaMalloc((void **)&d_fConList.x, 192*MaxNoofC180s*sizeof(float))) return -1;
  if ( cudaSuccess != cudaMalloc((void **)&d_fConList.y, 192*MaxNoofC180s*sizeof(float))) return -1;
  if ( cudaSuccess != cudaMalloc((void **)&d_fConList.z, 192*MaxNoofC180s*sizeof(float))) return -1;
  if ( cudaSuccess != cudaMalloc((void **)&d_fDisList.x, 192*MaxNoofC180s*sizeof(float))) return -1;
  if ( cudaSuccess != cudaMalloc((void **)&d_fDisList.y, 192*MaxNoofC180s*sizeof(float))) return -1;
  if ( cudaSuccess != cudaMalloc((void **)&d_fDisList.z, 192*MaxNoofC180s*sizeof(float))) return -1;
  if ( cudaSuccess != cudaMalloc((void **)&d_fRanList.x, 192*MaxNoofC180s*sizeof(float))) return -1;
  if ( cudaSuccess != cudaMalloc((void **)&d_fRanList.y, 192*MaxNoofC180s*sizeof(float))) return -1;
  if ( cudaSuccess != cudaMalloc((void **)&d_fRanList.z, 192*MaxNoofC180s*sizeof(float))) return -1;
  

  cudaMemset(d_C180_nn, 0, 3*192*sizeof(int));
  cudaMemset(d_C180_sign, 0, 180*sizeof(int));

  cudaMemset(d_XP, 0, 192*MaxNoofC180s*sizeof(float));
  cudaMemset(d_YP, 0, 192*MaxNoofC180s*sizeof(float));
  cudaMemset(d_ZP, 0, 192*MaxNoofC180s*sizeof(float));
  cudaMemset(d_X, 0, 192*MaxNoofC180s*sizeof(float));
  cudaMemset(d_Y, 0, 192*MaxNoofC180s*sizeof(float));
  cudaMemset(d_Z, 0, 192*MaxNoofC180s*sizeof(float));
  cudaMemset(d_XM, 0, 192*MaxNoofC180s*sizeof(float));
  cudaMemset(d_YM, 0, 192*MaxNoofC180s*sizeof(float));
  cudaMemset(d_ZM, 0, 192*MaxNoofC180s*sizeof(float));
  CudaErrorCheck();

  cudaMemset(d_CMx, 0, MaxNoofC180s*sizeof(float));
  cudaMemset(d_CMy, 0, MaxNoofC180s*sizeof(float));
  cudaMemset(d_CMz, 0, MaxNoofC180s*sizeof(float));
  CudaErrorCheck();

  cudaMemset(d_R0, 0, 3*192*sizeof(float));
  cudaMemset(d_pressList, 0, MaxNoofC180s*sizeof(float));
  cudaMemset(d_Youngs_mod, 0, MaxNoofC180s*sizeof(float));
  cudaMemset(d_bounding_xyz, 0, MaxNoofC180s*6*sizeof(float));
  cudaMemset(d_Minx, 0, 1024*sizeof(float));
  cudaMemset(d_Maxx, 0, 1024*sizeof(float));
  cudaMemset(d_Miny, 0, 1024*sizeof(float));
  cudaMemset(d_Maxy, 0, 1024*sizeof(float));
  cudaMemset(d_Minz, 0, 1024*sizeof(float));
  cudaMemset(d_Maxz, 0, 1024*sizeof(float));
  CudaErrorCheck();

  cudaMemset(d_velListX, 0, 192*MaxNoofC180s*sizeof(float));
  cudaMemset(d_velListY, 0, 192*MaxNoofC180s*sizeof(float));
  cudaMemset(d_velListZ, 0, 192*MaxNoofC180s*sizeof(float));


  cudaMemset(d_fConList.x, 0, 192*MaxNoofC180s*sizeof(float));
  cudaMemset(d_fConList.y, 0, 192*MaxNoofC180s*sizeof(float));
  cudaMemset(d_fConList.z, 0, 192*MaxNoofC180s*sizeof(float));

  cudaMemset(d_fDisList.x, 0, 192*MaxNoofC180s*sizeof(float));
  cudaMemset(d_fDisList.y, 0, 192*MaxNoofC180s*sizeof(float));
  cudaMemset(d_fDisList.z, 0, 192*MaxNoofC180s*sizeof(float));

  cudaMemset(d_fRanList.x, 0, 192*MaxNoofC180s*sizeof(float));
  cudaMemset(d_fRanList.y, 0, 192*MaxNoofC180s*sizeof(float));
  cudaMemset(d_fRanList.z, 0, 192*MaxNoofC180s*sizeof(float));

  cudaMemset(d_area, 0, MaxNoofC180s*sizeof(float));
  CudaErrorCheck();

  cudaMemset(d_boxMin, 0, 3*sizeof(float));
  cudaMemset(d_contactForces.x, 0, 192*MaxNoofC180s*sizeof(float));
  cudaMemset(d_contactForces.y, 0, 192*MaxNoofC180s*sizeof(float));
  cudaMemset(d_contactForces.z, 0, 192*MaxNoofC180s*sizeof(float));
  
  thrust::host_vector<angles3> theta0(192);
  thrust::device_vector<angles3> d_theta0V(192);
  angles3* d_theta0 = thrust::raw_pointer_cast(&d_theta0V[0]);

  h_contactForces.x = (float *)calloc(192*MaxNoofC180s, sizeof(float));
  h_contactForces.y = (float *)calloc(192*MaxNoofC180s, sizeof(float));
  h_contactForces.z = (float *)calloc(192*MaxNoofC180s, sizeof(float));

  thrust::device_vector<float> d_XMMV(MaxNoofC180s*192);
  thrust::fill(d_XMMV.begin(), d_XMMV.end(), 0.f);
  float *d_XMM = thrust::raw_pointer_cast(&d_XMMV[0]);
  
  thrust::device_vector<float> d_YMMV(MaxNoofC180s*192);
  thrust::fill(d_YMMV.begin(), d_YMMV.end(), 0.f);
  float *d_YMM = thrust::raw_pointer_cast(&d_YMMV[0]); 
  
  thrust::device_vector<float> d_ZMMV(MaxNoofC180s*192);
  thrust::fill(d_ZMMV.begin(), d_ZMMV.end(), 0.f);
  float *d_ZMM = thrust::raw_pointer_cast(&d_ZMMV[0]);

  thrust::device_vector<int> d_NoofNNlistV(1024*1024);
  thrust::fill(d_NoofNNlistV.begin(), d_NoofNNlistV.end(), 0);
  int *d_NoofNNlist = thrust::raw_pointer_cast(&d_NoofNNlistV[0]);

  thrust::device_vector<float> d_timeV(192*MaxNoofC180s);
  thrust::fill(d_timeV.begin(), d_timeV.end(), delta_t); 
  float *d_time = thrust::raw_pointer_cast(&d_timeV[0]);
  thrust::host_vector<float> h_timeV(192*MaxNoofC180s);

  thrust::host_vector<float> dt_listV(Time_steps + equiStepCount);
  thrust::fill(dt_listV.begin(), dt_listV.end(), delta_t);  
  thrust::device_vector<float> d_XtV(192*MaxNoofC180s);
  thrust::fill(d_XtV.begin(), d_XtV.end(), 0.f);
  float *d_Xt = thrust::raw_pointer_cast(&d_XtV[0]);

  thrust::device_vector<float> d_YtV(192*MaxNoofC180s);
  thrust::fill(d_YtV.begin(), d_YtV.end(), 0.f);
  float *d_Yt = thrust::raw_pointer_cast(&d_YtV[0]);

  thrust::device_vector<float> d_ZtV(192*MaxNoofC180s);
  thrust::fill(d_ZtV.begin(), d_ZtV.end(), 0.f);
  float *d_Zt = thrust::raw_pointer_cast(&d_ZtV[0]);

  thrust::device_vector<float> d_AdpErrorsV(192*MaxNoofC180s); 
  thrust::fill(d_AdpErrorsV.begin(), d_AdpErrorsV.end(), 0); 
  float *d_AdpErrors = thrust::raw_pointer_cast(&d_AdpErrorsV[0]); 

  thrust::host_vector<float> h_AdpErrors(192*MaxNoofC180s);

  thrust::device_vector<float> d_volumeV(MaxNoofC180s);
  thrust::host_vector<float> h_volume(MaxNoofC180s);
  thrust::fill(d_volumeV.begin(), d_volumeV.end(), 0.f);
  d_volume = thrust::raw_pointer_cast(&d_volumeV[0]);
  volume = thrust::raw_pointer_cast(&h_volume[0]);


  // thrust::device_vector<float> d_R0V(MaxNoofC180s);
  // thrust::host_vector<float> h_R0V(MaxNoofC180s);
  // d_R0 = thrust::raw_pointer_cast(&d_R0V[0]);
  // thrust::fill(h_R0V.begin(), h_R0V.end(), R0);
  // d_R0V = h_R0V;

  if (cudaSuccess != cudaMemcpy(d_R0, h_R0, 3*192*sizeof(float), cudaMemcpyHostToDevice)) return -1; 

  // velocities

  

  
  bounding_xyz = (float *)calloc(MaxNoofC180s*6, sizeof(float));
  CMx   = (float *)calloc(MaxNoofC180s, sizeof(float));
  CMy   = (float *)calloc(MaxNoofC180s, sizeof(float));
  CMz   = (float *)calloc(MaxNoofC180s, sizeof(float));
  //volume= (float *)calloc(MaxNoofC180s, sizeof(float));
  area= (float *)calloc(MaxNoofC180s, sizeof(float));
  cell_div = (char *)calloc(MaxNoofC180s, sizeof(char));
  cell_div_inds = (int *)calloc(MaxNoofC180s, sizeof(int));
  Minx  = (float *)calloc(1024, sizeof(float));
  Maxx  = (float *)calloc(1024, sizeof(float));
  Miny  = (float *)calloc(1024, sizeof(float));
  Maxy  = (float *)calloc(1024, sizeof(float));
  Minz  = (float *)calloc(1024, sizeof(float));
  Maxz  = (float *)calloc(1024, sizeof(float));
  NoofNNlist = (int *)calloc( 1024*1024,sizeof(int));
  NNlist =  (int *)calloc(32*1024*1024, sizeof(int));
  pressList = (float *)calloc(MaxNoofC180s, sizeof(float));
  resetIndices = (int *)calloc(MaxNoofC180s, sizeof(int));
  
  CPUMemory += MaxNoofC180s*7L*sizeof(float);
  CPUMemory += MaxNoofC180s*sizeof(float);
  CPUMemory += 3L*MaxNoofC180s*sizeof(float);
  CPUMemory += 6L*1024L*sizeof(float);
  CPUMemory += MaxNoofC180s*sizeof(char);
  CPUMemory += MaxNoofC180s*sizeof(int);
  CPUMemory += MaxNoofC180s*sizeof(int); 
  CPUMemory += 3*180*sizeof(float); 


  cudaMemcpy(d_pressList, pressList, MaxNoofC180s*sizeof(float), cudaMemcpyHostToDevice);

  velListX = (float *)calloc(192*MaxNoofC180s, sizeof(float)); 
  velListY = (float *)calloc(192*MaxNoofC180s, sizeof(float)); 
  velListZ = (float *)calloc(192*MaxNoofC180s, sizeof(float));

  

  

  // Better way to see how much GPU memory is being used.
  size_t totalGPUMem;
  size_t freeGPUMem;

  if ( cudaSuccess != cudaMemGetInfo ( &freeGPUMem, &totalGPUMem ) ) {
      printf("Couldn't read GPU Memory status\nExiting...\n");
      CudaErrorCheck();
      exit(1);
  }

  GPUMemory = totalGPUMem - freeGPUMem;


  cudaMemcpy(d_C180_nn,   C180_nn,   3*192*sizeof(int),cudaMemcpyHostToDevice);
  cudaMemcpy(d_C180_sign, C180_sign, 180*sizeof(int),cudaMemcpyHostToDevice);
  cudaMemcpy(d_C180_56,   C180_56,   7*92*sizeof(int),cudaMemcpyHostToDevice);

  cudaMemcpy(d_XP, X, 192*No_of_C180s*sizeof(float),cudaMemcpyHostToDevice);
  cudaMemcpy(d_YP, Y, 192*No_of_C180s*sizeof(float),cudaMemcpyHostToDevice);
  cudaMemcpy(d_ZP, Z, 192*No_of_C180s*sizeof(float),cudaMemcpyHostToDevice);
  cudaMemcpy(d_X,  X, 192*No_of_C180s*sizeof(float),cudaMemcpyHostToDevice);
  cudaMemcpy(d_Y,  Y, 192*No_of_C180s*sizeof(float),cudaMemcpyHostToDevice);
  cudaMemcpy(d_Z,  Z, 192*No_of_C180s*sizeof(float),cudaMemcpyHostToDevice);
  cudaMemcpy(d_XM, X, 192*No_of_C180s*sizeof(float),cudaMemcpyHostToDevice);
  cudaMemcpy(d_YM, Y, 192*No_of_C180s*sizeof(float),cudaMemcpyHostToDevice);
  cudaMemcpy(d_ZM, Z, 192*No_of_C180s*sizeof(float),cudaMemcpyHostToDevice);

  float tempX[192*No_of_C180s];


  cudaMemcpy(d_cell_div, cell_div, MaxNoofC180s*sizeof(char), cudaMemcpyHostToDevice);

  CudaErrorCheck();

  // Set the Youngs_mod for the cells
  youngsModArray = (float *)calloc(MaxNoofC180s, sizeof(float));
  for (int i =  0; i < MaxNoofC180s; ++i){
      youngsModArray[i] = stiffness1; 
  }
  
  if (useDifferentStiffnesses){
      printf("Making some cells softer... \n");
      if (closenessToCenter > 0.f && closenessToCenter < 1.f){
          printf("Only making cells within %f of max radius softer\n", closenessToCenter);
          CenterOfMass<<<No_of_C180s,256>>>(No_of_C180s,
                                            d_XP, d_YP, d_ZP,
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
                  youngsModArray[i] = stiffness2;
                  ++c; 
              }
          }

          printf("Made %d cells softer\n", c);
          
      } else {

          printf("Choosing softer cells randomly\n");
          int c = numberOfSofterCells;
          if (fractionOfSofterCells > 0.f && fractionOfSofterCells < 1.f){
              c = round(fractionOfSofterCells*(float)No_of_C180s);
          }

          if (c > No_of_C180s){
              printf("ERROR: Too many softer cells requested\n");
              return 12516;
          }

          if (useRigidSimulationBox){
              for (int i =0; i < c; ++i){
                  youngsModArray[i] = stiffness2; 
              }
          } else {
              float rands[1];
              for (int i =0; i < c; ++i){
                  ranmar(rands, 1);
                  int ind = round(rands[0]*No_of_C180s);
                  youngsModArray[ind] = stiffness2; 
              }
          }
          printf("Made %d cells softer\n", c);
      }
  }

  cudaMemcpy(d_Youngs_mod, youngsModArray, MaxNoofC180s*sizeof(float), cudaMemcpyHostToDevice);
  CudaErrorCheck();
  
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
      

      
      
      curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_MT19937);
      CudaErrorCheck();

      curandSetPseudoRandomGeneratorSeed(gen, time(NULL));
      CudaErrorCheck();

      if (rand_seed > 0){
          curandSetPseudoRandomGeneratorSeed(gen, rand_seed);
          CudaErrorCheck();
      }

      curandGenerate(gen, d_seeds, MaxNoofC180s*192);
      CudaErrorCheck();
  
      DeviceRandInit<<<(192*MaxNoofC180s)/256 + 1, 256>>>(d_rngStates, d_seeds, 192*MaxNoofC180s);
      CudaErrorCheck();
  }


  prevnoofblocks  = No_of_C180s;
  noofblocks      = No_of_C180s;
  threadsperblock = 192;
  printf("   no of blocks = %d, threadsperblock = %d, no of threads = %ld\n",
         noofblocks, threadsperblock, ((long) noofblocks)*((long) threadsperblock));

 // CenterOfMass<<<No_of_C180s,256>>>(No_of_C180s,
                                    d_X, d_Y, d_Z,
                                    d_CMx, d_CMy, d_CMz);
  
  //bounding_boxes<<<No_of_C180s,32>>>(No_of_C180s,d_X,d_Y,d_Z,
  //                                   d_bounding_xyz, d_CMx, d_CMy, d_CMz);

  //CudaErrorCheck(); 

  reductionblocks = (No_of_C180s-1)/1024+1;
  //minmaxpre<<<reductionblocks,1024>>>( No_of_C180s, d_bounding_xyz,
  //                                     d_Minx, d_Maxx, d_Miny, d_Maxy, d_Minz, d_Maxz);
 // CudaErrorCheck(); 
  //minmaxpost<<<1,1024>>>(reductionblocks, d_Minx, d_Maxx, d_Miny, d_Maxy, d_Minz, d_Maxz);
 // CudaErrorCheck(); 
 // cudaMemcpy(Minx, d_Minx, 6*sizeof(float),cudaMemcpyDeviceToHost);
  //  DL = 3.8f;
  
  globalrank = 0;


  // open trajectory file
  trajfile = fopen (trajFileName, "w");
  if ( trajfile == NULL)
  {
      printf("Failed to open %s \n", trajFileName);
      return -1;
  }

  FILE* forceFile = fopen(forces_file, "w");

  FILE* velFile = fopen("velocity2.xyz", "w");

#ifdef OUTPUT_ADP_ERROR
  FILE* timeFile = fopen("times", "w");
  FILE* errFile = fopen("errors", "w"); 
#endif 


  if (correct_com == true){
      CenterOfMass<<<No_of_C180s,256>>>(No_of_C180s,
                                        d_X, d_Y, d_Z,
                                        d_CMx, d_CMy, d_CMz);
      cudaMemcpy(CMx, d_CMx, No_of_C180s*sizeof(float),cudaMemcpyDeviceToHost);
      cudaMemcpy(CMy, d_CMy, No_of_C180s*sizeof(float),cudaMemcpyDeviceToHost);
      cudaMemcpy(CMz, d_CMz, No_of_C180s*sizeof(float),cudaMemcpyDeviceToHost);
      calc_sys_CM();
      //printf("sysCMx = %f, sysCMy = %f, sysCmz = %f\n", sysCMx, sysCMy, sysCMz);
      CorrectCoMMotion<<<(No_of_C180s*192)/1024 + 1, 1024>>>(d_X, d_Y, d_Z,
                                                             sysCMx, sysCMy, sysCMz,
                                                             No_of_C180s*192);
          
      CorrectCoMMotion<<<(No_of_C180s*192)/1024 + 1, 1024>>>(d_XP, d_YP, d_ZP,
                                                             sysCMx, sysCMy, sysCMz,
                                                             No_of_C180s*192);

      CorrectCoMMotion<<<(No_of_C180s*192)/1024 + 1, 1024>>>(d_XM, d_YM, d_ZM,
                                                             sysCMx, sysCMy, sysCMz,
                                                             No_of_C180s*192);
      CudaErrorCheck(); 
  }
  
  

  // Set up walls if needed
  if (useWalls == 1){
      // First we must make sure that the walls surround the
      // starting system.
      CenterOfMass<<<No_of_C180s,256>>>(No_of_C180s,
                                        d_X, d_Y, d_Z,
                                        d_CMx, d_CMy, d_CMz);
      cudaMemcpy(CMx, d_CMx, No_of_C180s*sizeof(float), cudaMemcpyDeviceToHost);
      cudaMemcpy(CMy, d_CMy, No_of_C180s*sizeof(float), cudaMemcpyDeviceToHost);
      cudaMemcpy(CMz, d_CMz, No_of_C180s*sizeof(float), cudaMemcpyDeviceToHost);
      float COMx = 0, COMy = 0, COMz = 0;

      for(int cell = 0; cell < No_of_C180s; cell++){
          COMx += CMx[cell];
          COMy += CMy[cell];
          COMz += CMz[cell];
      }

      COMx = COMx/No_of_C180s;
      COMy = COMy/No_of_C180s;
      COMz = COMz/No_of_C180s;


      if (perpAxis[0] == 'Z' || perpAxis[0] == 'z' ){
          // Check that the walls are far enough from the beginning cells
          float tempZ[192*No_of_C180s];
          cudaMemcpy(tempZ, d_Z, 192*No_of_C180s*sizeof(float), cudaMemcpyDeviceToHost);
          std::sort(tempZ, tempZ+No_of_C180s);
          float radius = 3.0 * divVol / 4.0;
          radius = radius/3.14159;
          radius = std::pow(radius, 0.33333333333);
          dAxis = dAxis * 2 * radius;

          if (dAxis < (tempZ[No_of_C180s] - tempZ[0])){
                  printf("Distance between walls is too small\nExiting...");
                  printf("Starting system size= %f\nWall gap = %f",
                         tempZ[No_of_C180s] - tempZ[0], dAxis);
                  return(-1);
              }

          wall1 = COMz - (dAxis/2.0);
          wall2 = COMz + (dAxis/2.0);
          wallLStart = COMx - (wallLen/2.0);
          wallLEnd = COMx + (wallLen/2.0);
          wallWStart = COMy - (wallWidth/2.0);
          wallWEnd = COMy + (wallWidth/2.0);
      }
      else {
          printf(" Invalid wall axis selection %s\nExiting...", perpAxis);
          return(-1);
      }

  }


  // Initialize pressures

  for (int cell = 0; cell < No_of_C180s; cell++){
      pressList[cell] = minPressure;
  }

  cudaMemcpy(d_pressList, pressList, No_of_C180s*sizeof(float), cudaMemcpyHostToDevice);
  CudaErrorCheck();

  float rGrowth = 0;
  bool growthDone = false;
  
  boxMin[0] = 0;
  boxMin[1] = 0;
  boxMin[2] = 0;
  
  // Setup simulation box, if needed (non-pbc)
  if (useRigidSimulationBox){
      printf("   Setup rigid (non-PBC) box...\n"); 
     // boxLength = boxLength*ceil(max( (Minx[5]-Minx[4]), max( (Minx[1]-Minx[0]), (Minx[3]-Minx[2]) ) ));
      //if (boxLength < minBoxLength) boxLength = minBoxLength
      //if (Side_length < 5) boxLength = boxLength * 5; 
      boxMin[0] = 0;
      boxMin[1] = 0;
      boxMin[2] = 0;
 
      DL = 2.9f;
      Xdiv = (int)((boxMax.x - boxMin[0])/DL+1);
      Ydiv = (int)((boxMax.y - boxMin[1])/DL+1);
      Zdiv = (int)((boxMax.z - boxMin[2])/DL+1);
     
      printf("   Done!\n");
      printf("   Simulation box minima:\n   X: %f, Y: %f, Z: %f\n", boxMin[0], boxMin[1], boxMin[2]);
      printf("   Simulation box maximum:\n   X: %f, Y: %f, Z: %f\n", boxMax.x, boxMax.y, boxMax.z);
     // printf("   Simulation box length = %f\n", boxLength);
  }

  CenterOfMass<<<No_of_C180s,256>>>(No_of_C180s, d_X, d_Y, d_Z, d_CMx, d_CMy, d_CMz);
  //DL = divVol; 
  CudaErrorCheck(); 
  makeNNlist<<<No_of_C180s/512+1,512>>>( No_of_C180s, d_CMx, d_CMy, d_CMz , Minx[0], Minx[2], Minx[4],
                                         attraction_range, Xdiv, Ydiv, Zdiv, d_NoofNNlist, d_NNlist, DL);
  CudaErrorCheck(); 


  cudaMemcpy(d_boxMin, boxMin, 3*sizeof(float), cudaMemcpyHostToDevice);
  CudaErrorCheck(); 


  // Code to set up pbc things
  if (usePBCs){
      boxLength = ceil(max( (Minx[5]-Minx[4]), max( (Minx[1]-Minx[0]), (Minx[3]-Minx[2]) ) ));
      //if (Side_length < 5) boxLength = boxLength * 5; 
      boxMin[0] = 0;
      boxMin[1] = 0;
      boxMin[2] = 0;
      
  }
  if (constrainAngles){
      // Code to initialize equillibrium angles
      float3 p, ni, nj, nk;
      for (int n = 0; n<180; n++){
          p = make_float3(X[n], Y[n], Z[n]); 

          ni = make_float3(X[C180_nn[0*192 + n]], Y[C180_nn[0*192 + n]], 
                           Z[C180_nn[0*192 + n]]); 
          
          nj = make_float3(X[C180_nn[1*192 + n]], Y[C180_nn[1*192 + n]], 
                           Z[C180_nn[1*192 + n]]);
          
          nk = make_float3(X[C180_nn[2*192 + n]], Y[C180_nn[2*192 + n]],
                           Z[C180_nn[2*192 + n]]);

          ni = ni-p;
          nj = nj-p;
          nk = nk-p; 

          theta0[n].aij = acos(dot(ni, nj)/(mag(ni)*mag(nj)));
          
          theta0[n].ajk = acos(dot(nj, nk)/(mag(nj)*mag(nk)));
          
          theta0[n].aik = acos(dot(ni, nk)/(mag(ni)*mag(nk)));

        }

      d_theta0V = theta0; 
      CudaErrorCheck(); 
  }

  // if (useDifferentStiffnesses && recalc_r0){

//       CalculateR0<<<No_of_C180s/1024 + 1, 1024>>>(d_R0,
//                                                   d_X, d_Y, d_Z,
//                                                   d_C180_nn,
//                                                   d_Youngs_mod,
//                                                   stiffness2,
//                                                   No_of_C180s);
// #ifdef RO_DEBUG
//       h_R0V = d_R0V;

//       cudaMemcpy(youngsModArray, d_Youngs_mod, sizeof(float)*MaxNoofC180s, cudaMemcpyDeviceToHost);
      
//       for (int i =0; i < No_of_C180s; ++i){
//           std::cout << "Cell " << i << " R0 = "
//                     << h_R0V[i] << " E = " << youngsModArray[i] << std::endl;
//       }
// #endif
  
//   }

  // Different kind of pressure stuff

  float r_CM_o = pow((3.0/4.0) * (1/3.14159) * divVol*2.0, 1.0/3);



  printf("   Total amount of GPU memory used =    %8.2lf MB\n",GPUMemory/(1024*1024.0));
  printf("   Total amount of CPU memory used =    %8.2lf MB\n",CPUMemory/(1024*1024.0));

  bool phase = false;

  // initial conditions

  // X[10] *= 1.1;

  // cudaMemcpy(d_X, X, 192*No_of_C180s*sizeof(float), cudaMemcpyHostToDevice);

  CalculateConForce<<<No_of_C180s,threadsperblock>>>( No_of_C180s, d_C180_nn, d_C180_sign,
                                                     d_X,  d_Y,  d_Z,
                                                     d_CMx, d_CMy, d_CMz,
                                                     d_R0, d_pressList, d_Youngs_mod , stiffness1, 
                                                     internal_damping, d_time, d_bounding_xyz,
                                                     attraction_strength, attraction_range,
                                                     repulsion_strength, repulsion_range,
                                                     viscotic_damping, mass,
                                                     Minx[0], Minx[2], Minx[4], Xdiv, Ydiv, Zdiv, d_NoofNNlist, d_NNlist, DL, gamma_visc,
                                                     wall1, wall2,
                                                     threshDist, useWalls,
                                                     d_velListX, d_velListY, d_velListZ,
                                                     useRigidSimulationBox, boxLength, d_boxMin, Youngs_mod,
                                                     constrainAngles, d_theta0, d_fConList, r_CM_o, boxMax, d_contactForces, d_volume, divVol); 
  CudaErrorCheck();

  CalculateDisForce<<<No_of_C180s, threadsperblock>>>(No_of_C180s, d_C180_nn, d_C180_sign, 
                                                     d_X, d_Y, d_Z,
                                                     internal_damping,
                                                     d_bounding_xyz,
                                                     attraction_range,
                                                     viscotic_damping,
                                                     Minx[0], Minx[2], Minx[4], Xdiv, Ydiv, Zdiv,
                                                     d_NoofNNlist, d_NNlist, DL, gamma_visc,
                                                     d_velListX, d_velListY, d_velListZ,
                                                     d_fDisList);
  CudaErrorCheck();


  if (binaryOutput){
      int t = MaxNoofC180s;
      fwrite(&t, sizeof(int), 1, trajfile);
      
      t = (int)useDifferentStiffnesses;
      fwrite(&t, sizeof(int), 1, trajfile);
      
      t = (Time_steps+equiStepCount+1) / trajWriteInt;
      fwrite(&t, sizeof(int), 1, trajfile);
      
    
      WriteBinaryTraj(0, trajfile, 1); 
  } else {
      fprintf(trajfile, "Header Start:\n");
      fprintf(trajfile, "Maximum number of cells:\n%d\n", MaxNoofC180s);

      fprintf(trajfile, "Using variable stiffness:\n");
      if (useDifferentStiffnesses) 
          fprintf(trajfile, "True\n");
      else
          fprintf(trajfile, "False\n");

      fprintf(trajfile, "Maximum number of frames:\n%d\n", (Time_steps+equiStepCount+1) / trajWriteInt);
      fprintf(trajfile, "Header End\n");
      write_traj(0, trajfile);
  }

  if (write_cont_force){
      fprintf(forceFile, "step,num_cells,cell_ind,node_ind,glob_node_ind,FX,FY,FZ,F,VX,VY,VZ,V,X,Y,Z,P,Vol\n");
      
      cudaMemcpy(h_contactForces.x, d_contactForces.x, 192*No_of_C180s*sizeof(float), cudaMemcpyDeviceToHost);
      cudaMemcpy(h_contactForces.y, d_contactForces.y, 192*No_of_C180s*sizeof(float), cudaMemcpyDeviceToHost);
      cudaMemcpy(h_contactForces.z, d_contactForces.z, 192*No_of_C180s*sizeof(float), cudaMemcpyDeviceToHost);
      cudaMemcpy(pressList, d_pressList, No_of_C180s*sizeof(float), cudaMemcpyDeviceToHost);
      
      cudaMemcpy(velListX, d_velListX, 192*No_of_C180s*sizeof(float), cudaMemcpyDeviceToHost);
      cudaMemcpy(velListY, d_velListY, 192*No_of_C180s*sizeof(float), cudaMemcpyDeviceToHost);
      cudaMemcpy(velListZ, d_velListZ, 192*No_of_C180s*sizeof(float), cudaMemcpyDeviceToHost);

      cudaMemcpy(volume, d_volume, No_of_C180s*sizeof(float), cudaMemcpyDeviceToHost);
      writeForces(forceFile, 0, No_of_C180s);
  }

  //return 0;


  int numNodes = No_of_C180s*192;
  // Simulation loop
  for ( step = 1; step < Time_steps+1 + equiStepCount; step++)
  {
      //printf("step %d\n", step);
      numNodes = No_of_C180s*192;
      Integrate<<<No_of_C180s, threadsperblock>>>(d_XP, d_YP, d_ZP,
                                                 d_X, d_Y, d_Z,
                                                 d_XM, d_YM, d_ZM, 
                                                 d_velListX, d_velListY, d_velListZ, 
                                                 d_time, mass,
                                                 d_fConList, d_fDisList, d_fRanList,
                                                 No_of_C180s, add_rands, d_rngStates, rand_scale_factor);
      CudaErrorCheck();

      ForwardTime<<<No_of_C180s, threadsperblock>>>(d_XP, d_YP, d_ZP, 
                                                   d_X , d_Y , d_Z ,
                                                   d_XM, d_YM, d_ZM,
                                                   No_of_C180s);
      CudaErrorCheck();


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
      PressureUpdate <<<No_of_C180s/1024 + 1, 1024>>> (d_pressList, minPressure, maxPressure, rGrowth, No_of_C180s,
                                                       useDifferentStiffnesses, stiffness1, d_Youngs_mod, step,
                                                       phase_count);
      CudaErrorCheck(); 
      
      if ( (step)%1000 == 0)
      {
          printf("   time %-8d %d cells, rGrowth %f, maxPop %f\n",step,No_of_C180s, rGrowth, maxPop);
      }


#ifdef FORCE_DEBUG
      printf("time %d  pressure = %f\n", step, Pressure);
#endif

      CalculateConForce<<<No_of_C180s,threadsperblock>>>( No_of_C180s, d_C180_nn, d_C180_sign,
                                                      d_X,  d_Y,  d_Z,
                                                      d_CMx, d_CMy, d_CMz,
                                                      d_R0, d_pressList, d_Youngs_mod , stiffness1, 
                                                      internal_damping, d_time, d_bounding_xyz,
                                                      attraction_strength, attraction_range,
                                                      repulsion_strength, repulsion_range,
                                                      viscotic_damping, mass,
                                                      Minx[0], Minx[2], Minx[4], Xdiv, Ydiv, Zdiv, d_NoofNNlist, d_NNlist, DL, gamma_visc,
                                                      wall1, wall2,
                                                      threshDist, useWalls,
                                                      d_velListX, d_velListY, d_velListZ,
                                                      useRigidSimulationBox, boxLength, d_boxMin, Youngs_mod,
                                                      constrainAngles, d_theta0, d_fConList, r_CM_o, boxMax, d_contactForces, d_volume, divVol); 
      CudaErrorCheck();


      CalculateDisForce<<<No_of_C180s, threadsperblock>>>(No_of_C180s, d_C180_nn, d_C180_sign, 
                                                         d_X, d_Y, d_Z,
                                                         internal_damping,
                                                         d_bounding_xyz,
                                                         attraction_range,
                                                         viscotic_damping,
                                                         Minx[0], Minx[2], Minx[4], Xdiv, Ydiv, Zdiv,
                                                         d_NoofNNlist, d_NNlist, DL, gamma_visc,
                                                         d_velListX, d_velListY, d_velListZ,
                                                         d_fDisList);
      CudaErrorCheck();

      // Calculate random Force here...
      if (add_rands){
          CalculateRanForce<<<No_of_C180s, threadsperblock>>>(No_of_C180s, d_rngStates, rand_scale_factor,
                                                              d_fRanList);
          CudaErrorCheck();
      }
      
      VelocityUpdateA<<<No_of_C180s, threadsperblock>>>(d_velListX, d_velListY, d_velListZ,
                                                        d_fConList, d_fRanList, delta_t, numNodes, mass);
      CudaErrorCheck();


      // Dissipative velocity update part...
      for (int s = 0; s < 1; ++s){
          VelocityUpdateB<<<No_of_C180s, threadsperblock>>>(d_velListX, d_velListY, d_velListZ,
                                                           d_fDisList, delta_t, numNodes, mass);
          CudaErrorCheck();
          
          CalculateDisForce<<<No_of_C180s, threadsperblock>>>(No_of_C180s, d_C180_nn, d_C180_sign, 
                                                             d_X, d_Y, d_Z,
                                                             internal_damping,
                                                             d_bounding_xyz,
                                                             attraction_range,
                                                             viscotic_damping,
                                                             Minx[0], Minx[2], Minx[4], Xdiv, Ydiv, Zdiv,
                                                             d_NoofNNlist, d_NNlist, DL, gamma_visc,
                                                             d_velListX, d_velListY, d_velListZ,
                                                             d_fDisList);
          CudaErrorCheck();
      }

      CenterOfMass<<<No_of_C180s,256>>>(No_of_C180s,
                                        d_X, d_Y, d_Z,
                                        d_CMx, d_CMy, d_CMz);
      CudaErrorCheck();
      if (step <= Time_steps && rGrowth > 0){
        // ------------------------------ Begin Cell Division ------------------------------------------------


        volumes<<<No_of_C180s,192>>>(No_of_C180s, d_C180_56,
                                     d_X, d_Y, d_Z,
                                     d_CMx , d_CMy, d_CMz,
                                     d_volume, d_cell_div, divVol,
                                     checkSphericity, d_area, phase_count, step,
                                     stiffness1, useDifferentStiffnesses, d_Youngs_mod,
                                     recalc_r0);
        CudaErrorCheck();

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

        count_and_get_div();

#ifdef PRINT_VOLUMES
        if (num_cell_div > 0){
            printf("Dividing cells: ");
            for (int i = 0; i<num_cell_div; i++){
                printf("%d ", cell_div_inds[i]);
            }
          
            printf("\n");
        }
#endif
        for (int divCell = 0; divCell < num_cell_div; divCell++) {
          globalrank = cell_div_inds[divCell];
          float norm[3];

          norm[0] = 0; 
          norm[1] = 1; 
          norm[2] = 0;
          
          if (useDivPlaneBasis)
              GetRandomVectorBasis(norm, divPlaneBasis);
          else
              GetRandomVector(norm);

#ifdef TURNOFF_RAN
          norm[0] = 0; 
          norm[1] = 1; 
          norm[2] = 0;
#endif
          
          cudaMemcpy( d_ran2, norm, 3*sizeof(float), cudaMemcpyHostToDevice);
          CudaErrorCheck();
          
          NDIV[globalrank] += 1;

          cell_division<<<1,256>>>(globalrank,
                                   d_XP, d_YP, d_ZP, 
                                   d_X, d_Y, d_Z,
                                   d_XM, d_YM, d_ZM, 
                                   d_CMx, d_CMy, d_CMz,
                                   d_velListX, d_velListY, d_velListZ, 
                                   No_of_C180s, d_ran2, repulsion_range);
          CudaErrorCheck()
          resetIndices[divCell] = globalrank;
          resetIndices[divCell + num_cell_div] = No_of_C180s;

          if (daughtSameStiffness){
              youngsModArray[No_of_C180s] = youngsModArray[globalrank];
              
          } else {
              youngsModArray[No_of_C180s] = stiffness1; 
          }
          
          ++No_of_C180s;
          if (No_of_C180s > MaxNoofC180s){
              printf("ERROR: Population is %d, only allocated enough memory for %d\n",
                     No_of_C180s, MaxNoofC180s);
              printf("ERROR: Fatal error, crashing...\n");
              return -69;
          }
        }
        
        if (num_cell_div>0){
            cudaMemcpy(d_resetIndices, resetIndices, 2*num_cell_div*sizeof(int),
                       cudaMemcpyHostToDevice);

            CudaErrorCheck(); 

            PressureReset <<<(2*num_cell_div)/512 + 1, 512>>> (d_resetIndices, d_pressList, minPressure, 2*num_cell_div); 
            CudaErrorCheck();

            cudaMemcpy(d_Youngs_mod, youngsModArray,
                       sizeof(float)*No_of_C180s, cudaMemcpyHostToDevice);
            CudaErrorCheck();

        }

        if (countOnlyInternal == 1){
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
        // --------------------------------------- End Cell Division -----------
      }

      // ----------------------------------------- Begin Cell Death ------------

      // Placeholder************************************************************

      // ----------------------------------------- End Cell Death --------------

      bounding_boxes<<<No_of_C180s,32>>>(No_of_C180s,
                                         d_X,d_Y,d_Z,
                                         d_bounding_xyz, d_CMx, d_CMy, d_CMz);
      CudaErrorCheck();
      
      

          
      reductionblocks = (No_of_C180s-1)/1024+1;
      minmaxpre<<<reductionblocks,1024>>>( No_of_C180s, d_bounding_xyz,
                                           d_Minx, d_Maxx, d_Miny, d_Maxy, d_Minz, d_Maxz);
      CudaErrorCheck(); 

      minmaxpost<<<1,1024>>>( reductionblocks, d_Minx, d_Maxx, d_Miny, d_Maxy, d_Minz, d_Maxz);
      
      CudaErrorCheck(); 

      cudaMemset(d_NoofNNlist, 0, 1024*1024);

      cudaMemcpy(Minx, d_Minx, 6*sizeof(float), cudaMemcpyDeviceToHost);
      Xdiv = (int)((Minx[1]-Minx[0])/DL+1);
      Ydiv = (int)((Minx[3]-Minx[2])/DL+1);
      Zdiv = (int)((Minx[5]-Minx[4])/DL+1);

      makeNNlist<<<No_of_C180s/512+1,512>>>( No_of_C180s, d_bounding_xyz, Minx[0], Minx[2], Minx[4],
                                             attraction_range, Xdiv, Ydiv, Zdiv, d_NoofNNlist, d_NNlist, DL);
      CudaErrorCheck();

      if (!growthDone && step > Time_steps+1){
          printf("Cell growth halted.\nProceeding with MD simulation without growth...\n");
          growthDone = true;
          
          if (useDifferentStiffnesses && !duringGrowth){
              printf("Now making some cells softer...\n");
              int softCellCounter = 0;
              if (fractionOfSofterCells > 0){
                  numberOfSofterCells = roundf(fractionOfSofterCells*No_of_C180s); 
              }

              printf("Will make %d cells softer\n", numberOfSofterCells); 
              
              if (chooseRandomCellIndices){
                  float rnd[1];
                  //int* chosenIndices = (int*)malloc(numberOfSofterCells, sizeof(int));
                  int chosenIndices[numberOfSofterCells]; 
                  
                  for (int i = 0; i < numberOfSofterCells; i++){
                      chosenIndices[i] = -1; 
                  }
                  
                  bool indexChosen = false;
                  int cellInd = -1;

                  printf("Make cells with indices "); 
                  
                  while (softCellCounter < numberOfSofterCells){
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

                  for (int i = 0; i < numberOfSofterCells; i++){
                      youngsModArray[chosenIndices[i]] = stiffness2; 
                  }
              }
              else {
                  // search for the oldest cells near the center of the system, and make them soft
                  cudaMemcpy(CMx, d_CMx, No_of_C180s*sizeof(float),cudaMemcpyDeviceToHost);
                  cudaMemcpy(CMy, d_CMy, No_of_C180s*sizeof(float),cudaMemcpyDeviceToHost);
                  cudaMemcpy(CMz, d_CMz, No_of_C180s*sizeof(float),cudaMemcpyDeviceToHost);

                  float Rmax2 = getRmax2();
                  float R2, dx, dy, dz;
                  int cellInd = 0; 
                  calc_sys_CM();

                  float f = 1 - closenessToCenter;
              
                  printf("Made cells with indices "); 

                  while (softCellCounter < numberOfSofterCells && cellInd < No_of_C180s){
                      dx = CMx[cellInd] - sysCMx; 
                      dy = CMy[cellInd] - sysCMy; 
                      dz = CMz[cellInd] - sysCMz;

                      R2 = dx*dx + dy*dy + dz*dz;

                      if (R2 <= f*f*Rmax2){
                          printf("%d, ", cellInd); 
                          softCellCounter++; 
                          youngsModArray[cellInd] = stiffness2; 

                      }
                      cellInd++; 
                  }
              }
              
              cudaMemcpy(d_Youngs_mod, youngsModArray, No_of_C180s*sizeof(float), cudaMemcpyHostToDevice);
              printf("\b\b softer\n"); 
          }

      }

      if (correct_com == true){
          cudaMemcpy(CMx, d_CMx, No_of_C180s*sizeof(float),cudaMemcpyDeviceToHost);
          cudaMemcpy(CMy, d_CMy, No_of_C180s*sizeof(float),cudaMemcpyDeviceToHost);
          cudaMemcpy(CMz, d_CMz, No_of_C180s*sizeof(float),cudaMemcpyDeviceToHost);
          calc_sys_CM();
          //printf("sysCMx = %f, sysCMy = %f, sysCmz = %f\n", sysCMx, sysCMy, sysCMz);
          CorrectCoMMotion<<<(No_of_C180s*192)/1024 + 1, 1024>>>(d_X, d_Y, d_Z,
                                                                 sysCMx, sysCMy, sysCMz,
                                                                 No_of_C180s*192);
          CudaErrorCheck(); 
      }

      if ( step%trajWriteInt == 0 )
      {
          //printf("   Writing trajectory to traj.xyz...\n");
          frameCount++; 
          cudaMemcpy(X, d_X, 192*No_of_C180s*sizeof(float),cudaMemcpyDeviceToHost);
          cudaMemcpy(Y, d_Y, 192*No_of_C180s*sizeof(float),cudaMemcpyDeviceToHost);
          cudaMemcpy(Z, d_Z, 192*No_of_C180s*sizeof(float),cudaMemcpyDeviceToHost);
          
          if (binaryOutput)
              WriteBinaryTraj(step, trajfile, frameCount);
          else
              write_traj(step, trajfile);

          if (write_cont_force == true){

              cudaMemcpy(h_contactForces.x, d_contactForces.x, 192*No_of_C180s*sizeof(float), cudaMemcpyDeviceToHost);
              cudaMemcpy(h_contactForces.y, d_contactForces.y, 192*No_of_C180s*sizeof(float), cudaMemcpyDeviceToHost);
              cudaMemcpy(h_contactForces.z, d_contactForces.z, 192*No_of_C180s*sizeof(float), cudaMemcpyDeviceToHost);
              
              cudaMemcpy(velListX, d_velListX, 192*No_of_C180s*sizeof(float), cudaMemcpyDeviceToHost);
              cudaMemcpy(velListY, d_velListY, 192*No_of_C180s*sizeof(float), cudaMemcpyDeviceToHost);
              cudaMemcpy(velListZ, d_velListZ, 192*No_of_C180s*sizeof(float), cudaMemcpyDeviceToHost);
              
              cudaMemcpy(X, d_X, 192*No_of_C180s, cudaMemcpyDeviceToHost);
              cudaMemcpy(Y, d_Y, 192*No_of_C180s, cudaMemcpyDeviceToHost);
              cudaMemcpy(Z, d_Z, 192*No_of_C180s, cudaMemcpyDeviceToHost);
              
              cudaMemcpy(pressList, d_pressList, No_of_C180s*sizeof(float), cudaMemcpyDeviceToHost);
              cudaMemcpy(volume, d_volume, No_of_C180s*sizeof(float), cudaMemcpyDeviceToHost);
                    
              writeForces(forceFile, step, No_of_C180s);
          }
      }

      myError = cudaGetLastError();
      if ( cudaSuccess != myError )
      {
          printf( "Error %d: %s!\n",myError,cudaGetErrorString(myError) );return(-1);
      }

      if (step > phase_count && phase){
          printf("In phase 2\n");
          phase = false;
          if (useDifferentStiffnesses && recalc_r0){
              CalculateR0<<<No_of_C180s/1024 + 1, 1024>>>(d_R0,
                                                          d_X, d_Y, d_Z,
                                                          d_C180_nn,
                                                          d_Youngs_mod,
                                                          stiffness2,
                                                          No_of_C180s);
#ifdef RO_DEBUG
              thrust::fill(h_R0V.begin(), h_R0V.end(), 0.f);
              h_R0V = d_R0V;
          
              cudaMemcpy(youngsModArray, d_Youngs_mod, sizeof(float)*MaxNoofC180s, cudaMemcpyDeviceToHost);
      
              for (int i =0; i < No_of_C180s; ++i){
                  std::cout << "Cell " << i << " R0 = "
                            << h_R0V[i] << " E = " << youngsModArray[i] << std::endl;
              }
#endif
          }
      }
  }

  if (binaryOutput){
      fseek(trajfile, 0, SEEK_SET);
      fwrite(&No_of_C180s, sizeof(int), 1, trajfile);
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

  cudaFree( (void *)d_bounding_xyz );
  cudaFree( (void *)d_XP );
  cudaFree( (void *)d_YP );
  cudaFree( (void *)d_ZP );
  cudaFree( (void *)d_X  );
  cudaFree( (void *)d_Y  );
  cudaFree( (void *)d_Z  );
  cudaFree( (void *)d_XM );
  cudaFree( (void *)d_YM );
  cudaFree( (void *)d_ZM );
  cudaFree( (void *)d_CMx );
  cudaFree( (void *)d_CMy );
  cudaFree( (void *)d_CMz );
  cudaFree( (void *)d_ran2 );

  cudaFree( (void *)d_C180_nn);
  cudaFree( (void *)d_C180_sign);
  cudaFree( (void *)d_cell_div);
  free(X); free(Y); free(Z);
  free(bounding_xyz);
  free(CMx); free(CMy); free(CMz);
  free(dividingCells); free(totalCells);
  free(NDIV);
  free(volume);
  free(Minx); free(Miny); free(Minz);
  free(Maxx); free(Maxy); free(Maxz);
  free(NoofNNlist);
  free(NNlist);
  free(ran2);
  free(num_new_cells_per_step);
  free(cell_div_inds);
  free(pressList);

  free(velListX); 
  free(velListY); 
  free(velListZ); 

  fclose(trajfile);
  fclose(forceFile);
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

  X = (float *)calloc(192*MaxNoofC180s,sizeof(float));
  Y = (float *)calloc(192*MaxNoofC180s,sizeof(float));
  Z = (float *)calloc(192*MaxNoofC180s,sizeof(float));

  bounding_xyz = (float *)calloc(MaxNoofC180s,6*sizeof(float));

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

  if (useRigidSimulationBox){
      float rCheck = powf(0.75*(1.f/3.14159)*0.786, 1.f/3.f); // this code is magical
      printf("Check radius = %f\n", rCheck);
      float3 allCMs[Orig_No_of_C180s];

      float vol = 0;
      int k = 0;
      
      vol = boxMax.x*boxMax.y*boxMax.z;
      k = floor(vol/0.786);
      
      if (k < Orig_No_of_C180s){
          fprintf(stderr, "ERROR: Simulation box is too small\n");
          fprintf(stderr, "       Big enough for %d\n", k);
          return 27;
      }

      printf("Can fit upto %d cells\n", k);

      int c = 0;
      float rands[3];

      if (rand_pos){
          while (true){
              ranmar(rands, 3);
              float3 CM = make_float3(rands[0]*(boxMax.x - 1.f)  + 1.f,
                                      rands[1]*(boxMax.y - 1.f)  + 1.f,
                                      0.f);
              if (flatbox == 1){
                  CM.z = boxMax.z/2;
              } else {
                  CM.z = rands[2]*(boxMax.z - 1.f)  + 1.f;
              }


              bool farEnough = true;
              
              farEnough = !(CM.x+rCheck > boxMax.x ||
                            CM.y+rCheck > boxMax.y ||
                            CM.z+rCheck > boxMax.z);
              
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

          for (int cellInd = 0; cellInd < Orig_No_of_C180s; ++cellInd){
              for(int nodeInd = 0; nodeInd < 180; ++nodeInd){
                  X[cellInd*192 + nodeInd] = initx[nodeInd] + allCMs[cellInd].x;
                  Y[cellInd*192 + nodeInd] = inity[nodeInd] + allCMs[cellInd].y;
                  Z[cellInd*192 + nodeInd] = initz[nodeInd] + allCMs[cellInd].z;
              }
          }
      } else {

          // crash because this doesn't work right now.
          // printf("Simulation in a box must be with random positions\n");
          // exit(12);

          rCheck *= 1.2;
          float3 center = 0.5*boxMax;

          for ( rank = 0; rank < Orig_No_of_C180s; ++rank )
          {
              ey=rank%Side_length;
              ex=rank/Side_length;

              for ( atom = 0 ; atom < 180 ; ++atom)
              {
                  X[rank*192+atom] = initx[atom] + L1*ex + 0.5*L1 + center.x;
                  Y[rank*192+atom] = inity[atom] + L1*ey + 0.5*L1 + center.y;
                  Z[rank*192+atom] = initz[atom] + center.z;
              }

          }
      }

      // check all the fucking positions...
      for (int i = 0; i < Orig_No_of_C180s*192; ++i){
          if (X[i] > boxMax.x ||
              Y[i] > boxMax.y ||
              Z[i] > boxMax.z){

              printf("shit is in the fan\n");
              printf("%f %f %f\n", X[i], Y[i], Z[i]);
              //exit(4); 
          }
                               
      }
  }
  else{
      for ( rank = 0; rank < Orig_No_of_C180s; ++rank )
      {
          ey=rank%Side_length;
          ex=rank/Side_length;

          for ( atom = 0 ; atom < 180 ; ++atom)
          {
              X[rank*192+atom] = initx[atom] + L1*ex + 0.5*L1;
              Y[rank*192+atom] = inity[atom] + L1*ey + 0.5*L1;
              Z[rank*192+atom] = initz[atom] + zOffset;
          }

      }
  }

  return(0);
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

  for (int i = 0; i < 180; ++i){
      int N1 = C180_nn[0 + i];
      int N2 = C180_nn[192 + i];
      int N3 = C180_nn[384 + i];

      float3 a, b; 
      a = make_float3(X[i], Y[i], Z[i]);

      b = make_float3(X[N1], Y[N1], Z[N1]);
      h_R0[0 + i] = mag(a-b);

      b = make_float3(X[N2], Y[N2], Z[N2]);
      h_R0[192 + i] = mag(a-b);

      b = make_float3(X[N3], Y[N3], Z[N3]);
      h_R0[384 + i] = mag(a-b);
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
        ranZOffset = coreParams["random_z_offset?"].asInt();
        zOffset = coreParams["z_offset"].asFloat();
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
        rMax = coreParams["growth_rate"].asFloat();
        checkSphericity = coreParams["checkSphericity"].asBool();
        constrainAngles = coreParams["constrainAngles"].asBool();
        dt_max = coreParams["dt_max"].asFloat();
        dt_tol = coreParams["dt_tol"].asFloat();
        doAdaptive_dt = coreParams["doAdaptive_dt"].asBool();
        phase_count = coreParams["phase_count"].asInt();
        write_cont_force = coreParams["write_cont_force"].asBool();
        std::strcpy(forces_file, coreParams["forces_file"].asString().c_str());
        correct_com = coreParams["correct_com"].asBool();
                                 
    }

    Json::Value countParams = inpRoot.get("counting", Json::nullValue);
    if (countParams == Json::nullValue){
        // countCells = FALSE;
        printf("ERROR: Cannot load counting parameters\nExiting");
        return -1;
    }
    else {
        // countCells = countParams["countcells"].asBool();
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

    Json::Value wallParams = inpRoot.get("walls", Json::nullValue);

    if (wallParams == Json::nullValue){
        printf("ERROR: Cannot load wall parameters\nExiting");
        return -1;
    }
    else{
        useWalls = wallParams["useWalls"].asInt();
        std::strcpy(perpAxis, wallParams["perpAxis"].asString().c_str());
        dAxis = wallParams["dAxis"].asFloat();
        wallLen = wallParams["wallLen"].asFloat();
        wallWidth = wallParams["wallWidth"].asFloat();
        threshDist = wallParams["threshDist"].asFloat();
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
    }

    Json::Value stiffnessParams = inpRoot.get("stiffnessParams", Json::nullValue);

    if (stiffnessParams == Json::nullValue){
        printf("ERROR: Cannot load stiffness parameters\n");
        return -1;
    } else {
        useDifferentStiffnesses = stiffnessParams["useDifferentStiffnesses"].asBool();
        stiffness2 = stiffnessParams["softStiffFactor"].asFloat() * Youngs_mod;
        numberOfSofterCells = stiffnessParams["numberOfSofterCells"].asInt();
        duringGrowth = stiffnessParams["duringGrowth"].asBool(); 
        closenessToCenter = stiffnessParams["closenessToCenter"].asFloat();
        startAtPop = stiffnessParams["startAtPop"].asInt();
        fractionOfSofterCells = stiffnessParams["fractionOfSofterCells"].asFloat();
        chooseRandomCellIndices = stiffnessParams["chooseRandomCellIndices"].asBool();
        daughtSameStiffness = stiffnessParams["daughtSameStiffness"].asBool();
        recalc_r0 = stiffnessParams["recalc_r0"].asBool();
    }

    Json::Value boxParams = inpRoot.get("boxParams", Json::nullValue);

    if (boxParams == Json::nullValue){
        printf("ERROR: Cannot load box parameters\n");
        return -1;
    } else{
        useRigidSimulationBox = boxParams["useRigidSimulationBox"].asBool();
        usePBCs = boxParams["usePBCs"].asBool();
        boxLength = boxParams["boxLength"].asFloat();
        boxMax.x = boxParams["box_len_x"].asFloat();
        boxMax.y = boxParams["box_len_y"].asFloat(); 
        boxMax.z = boxParams["box_len_z"].asFloat();
        flatbox = boxParams["flatbox"].asBool();
        rand_pos = boxParams["rand_pos"].asBool();
    }
    
    
    if (ranZOffset == 0)
        zOffset = 0.0;

    if (dt_tol > dt_max || dt_max <= 0 || dt_tol < 0){
        printf("ERROR: Invalid time step parameters\n");
        printf("FATAL ERROR\n");
        exit(-60); 
    }

    Json::Value randParams = inpRoot.get("rand_params", Json::nullValue);

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
    printf("      spring equilibrium  = %f\n",R0);
    printf("      repulsion range     = %f\n",repulsion_range);
    printf("      attraction range    = %f\n",attraction_range);
    printf("      repulsion strength  = %f\n",repulsion_strength);
    printf("      attraction strength = %f\n",attraction_strength);
    printf("      Youngs modulus      = %f\n",stiffness1);
    printf("      viscotic damping    = %f\n",viscotic_damping);
    printf("      internal damping    = %f\n",internal_damping);
    printf("      division volume     = %f\n",divVol);
    printf("      ran_z_offset?       = %d\n", ranZOffset);
    printf("      z_offset            = %f\n", zOffset);
    printf("      Time steps          = %d\n",Time_steps);
    printf("      delta t             = %f\n",delta_t);
    printf("      Restart             = %d\n",Restart);
    printf("      trajWriteInterval   = %d\n",trajWriteInt);
    printf("      countOnlyInternal   = %d\n", countOnlyInternal);
    printf("      radFrac             = %f\n", radFrac);
    printf("      newCellCountInt     = %d\n", newCellCountInt);
    printf("      equiStepCount       = %d\n", equiStepCount);
    printf("      trajFileName        = %s\n", trajFileName);
    printf("      doPopModel          = %d\n", doPopModel);
    printf("      totalFood           = %f\n", totalFood);
    printf("      cellFoodCons        = %f\n", cellFoodCons);
    printf("      cellFoodConsDiv     = %f\n", cellFoodConsDiv);
    printf("      cellFoodRel         = %f\n", cellFoodRel);
    printf("      useWalls            = %d\n", useWalls);
    printf("      perpAxis            = %s\n", perpAxis);
    printf("      dAxis               = %f\n", dAxis);
    printf("      wallLen             = %f\n", wallLen);
    printf("      wallWidth           = %f\n", wallWidth);
    printf("      thresDist           = %f\n", threshDist);
    printf("      maxPressure         = %f\n", maxPressure);
    printf("      minPressure         = %f\n", minPressure);
    printf("      growth_rate         = %f\n", rMax);
    printf("      checkSphericity     = %d\n", checkSphericity);
    printf("      gamma_visc          = %f\n", gamma_visc);
    printf("      useDivPlanebasis    = %d\n", useDivPlaneBasis);
    printf("      divPlaneBasisX      = %f\n", divPlaneBasis[0]);
    printf("      divPlaneBasisY      = %f\n", divPlaneBasis[1]);
    printf("      divPlaneBasisZ      = %f\n", divPlaneBasis[2]);
    printf("      useDifferentStiffnesses = %d\n", useDifferentStiffnesses);
    printf("      softYoungsMod       = %f\n", softYoungsMod);
    printf("      numberOfsofterCells = %d\n", numberOfSofterCells);
    printf("      duringGrowth        = %d\n", duringGrowth);
    printf("      closenesstoCenter   = %f\n", closenessToCenter);
    printf("      startAtPop          = %d\n", startAtPop);
    printf("      fractionOfSofterCells   = %f\n", fractionOfSofterCells);
    printf("      chooseRandomCellIndices = %d\n", chooseRandomCellIndices);
    printf("      daughtSameStiffness = %d\n", daughtSameStiffness);
    printf("      recalc_r0           = %d\n", recalc_r0);
    printf("      useRigidSimulationBox = %d\n", useRigidSimulationBox);
    printf("      usePBCs             = %d\n", usePBCs);
    printf("      boxLength           = %f\n", boxLength);
    printf("      box_len_x           = %f\n", boxMax.x);
    printf("      box_len_y           = %f\n", boxMax.y);
    printf("      box_len_z           = %f\n", boxMax.z);
    printf("      flatbox             = %d\n", flatbox); 
    printf("      doAdaptive_dt       = %d\n", doAdaptive_dt); 
    printf("      dt_max              = %f\n", dt_max); 
    printf("      dt_tol              = %f\n", dt_tol);
    printf("      add_rands           = %d\n", add_rands);
    printf("      rand_seed           = %d\n", rand_seed);
    printf("      rand_scale_factor   = %f\n", rand_scale_factor);
    printf("      phase_count         = %d\n", phase_count);
    printf("      correct_com         = %d\n", correct_com);
    
    

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

    if ( !(closenessToCenter >=0 && closenessToCenter <= 1) ){
        printf("ERROR: closenessToCenter is not in [0, 1]\n");
        printf("ERROR: invalid input parameter\n");
        return -1;
    }

    if (useWalls && useRigidSimulationBox){
        printf("ERROR: Cannot use infinite XY walls and rigid simulation box simultaneously.\n");
        printf("ERROR: Only use on or the other.\n");
        return -1;
    }

    if (fractionOfSofterCells > 1.0){
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
  if ( fscanf(infil,"%d",&useWalls)            != 1 ) {error = 28 ;}
  if ( fscanf(infil,"%s",perpAxis)             != 1 ) {error = 29 ;}
  if ( fscanf(infil,"%f",&dAxis)               != 1 ) {error = 30 ;}
  if ( fscanf(infil,"%f",&wallLen)             != 1 ) {error = 31 ;}
  if ( fscanf(infil,"%f",&wallWidth)           != 1 ) {error = 32 ;}
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
  printf("      spring equilibrium  = %f\n",R0);
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
  printf("      doPopModel          = %d\n", doPopModel);
  printf("      totalFood           = %f\n", totalFood);
  printf("      cellFoodCons        = %f\n", cellFoodCons);
  printf("      cellFoodConsDiv     = %f\n", cellFoodConsDiv);
  printf("      cellFoodRel         = %f\n", cellFoodRel);
  printf("      useWalls            = %d\n", useWalls);
  printf("      perpAxis            = %s\n", perpAxis);
  printf("      dAxis               = %f\n", dAxis);
  printf("      wallLen             = %f\n", wallLen);
  printf("      wallWidth           = %f\n", wallWidth);
  printf("      thresDist           = %f\n", threshDist);


  return(0);
}




//C *****************************************************************




void write_traj(int t_step, FILE* trajfile)
{

  fprintf(trajfile, "%d\n", No_of_C180s * 192);
  fprintf(trajfile, "Step: %d frame: %d\n", t_step, t_step/trajWriteInt);
  
  if (useDifferentStiffnesses){
      for (int c = 0; c < No_of_C180s; c++){
          if (youngsModArray[c] == stiffness1)
              fprintf(trajfile, "cell: %d H\n", c);
          else if(youngsModArray[c] == stiffness2)
              fprintf(trajfile, "cell: %d C\n", c);
          else
              fprintf(trajfile, "cell: %d UnknownStiffness\n", c);

          for (int p = 0; p < 192; p++)
          {
              fprintf(trajfile, "%.7f,  %.7f,  %.7f\n", X[(c*192)+p], Y[(c*192)+p], Z[(c*192)+p]);
          }
      }
        
  } else {
      for (int c = 0; c < No_of_C180s; c++){
              fprintf(trajfile, "cell: %d\n", c);
              
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
    if (useDifferentStiffnesses){
        int cellType = 0; 
        for (int c = 0; c < No_of_C180s; c++){
            fwrite(&c, sizeof(int), 1, trajFile);
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
            fwrite(&c, sizeof(int), 1, trajFile);
            
            fwrite(X + (c*192), sizeof(float), 192, trajFile); 
            fwrite(Y + (c*192), sizeof(float), 192, trajFile); 
            fwrite(Z + (c*192), sizeof(float), 192, trajFile); 
        }
    }
        
    
}

void write_vel(int t_step, FILE* velFile){
    fprintf(velFile, "%d\n", No_of_C180s * 192);
    fprintf(velFile, "Step: %d\n", t_step);
    for (int p = 0; p < No_of_C180s*192; p++)
    {
        fprintf(velFile, "%.7f,  %.7f,  %.7f\n", velListX[p], velListY[p], velListZ[p]);
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
}



inline void calc_sys_CM(){ // Put this into a kernel at some point

  sysCMx = 0;
  sysCMy = 0;
  sysCMz = 0;

  for (int cellInd = 0; cellInd < No_of_C180s; cellInd++) {
    sysCMx += CMx[cellInd];
    sysCMy += CMy[cellInd];
    sysCMz += CMz[cellInd];
  }

  sysCMx = sysCMx/No_of_C180s;
  sysCMy = sysCMy/No_of_C180s;
  sysCMz = sysCMz/No_of_C180s;

}


inline float getRmax2(){
  float dx, dy, dz, Rmax2 = 0;
  for (int cell = 0; cell < No_of_C180s; cell++) {
    dx = CMx[cell] - sysCMx;
    dy = CMy[cell] - sysCMy;
    dz = CMz[cell] - sysCMz;

    Rmax2 = max(Rmax2, dx*dx + dy*dy + dz*dz);

  }

  return Rmax2;

}

inline int num_cells_far(){

  if (num_cell_div == 0 || No_of_C180s < 50) return 0;

  cudaMemcpy(CMx, d_CMx, No_of_C180s*sizeof(float), cudaMemcpyDeviceToHost);
  cudaMemcpy(CMy, d_CMy, No_of_C180s*sizeof(float), cudaMemcpyDeviceToHost);
  cudaMemcpy(CMz, d_CMz, No_of_C180s*sizeof(float), cudaMemcpyDeviceToHost);

  calc_sys_CM();

  float dx, dy, dz, dr2;
  float Rmax2 = getRmax2();
  int farCellCount = 0;

  for (int cell = No_of_C180s - num_cell_div; cell < No_of_C180s; cell++) { // Only check the newest cells
    dx = CMx[cell] - sysCMx;
    dy = CMy[cell] - sysCMy;
    dz = CMz[cell] - sysCMz;

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
            fprintf(forceFile, "%d,%d,%d,%d,%d,%.8f,%.8f,%.8f,%.8f,%.8f,%.8f,%.8f,%.8f,%.8f,%.8f,%.8f,%.8f,%.8f\n",
                    t_step, num_cells, c, n, c*192+n,
                    h_contactForces.x[c*192 + n],
                    h_contactForces.y[c*192 + n],
                    h_contactForces.z[c*192 + n],
                    mag(make_float3(h_contactForces.x[c*192 + n],
                                    h_contactForces.y[c*192 + n],
                                    h_contactForces.z[c*192 + n])),
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
                    volume[c]
                );
                        
        }


    }
}


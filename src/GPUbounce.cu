//#define FORCE_DEBUG
//#define PRINT_VOLUMES
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <locale.h>
#include <algorithm>
#include <iostream>
#include <fstream>
#include <streambuf>
#include <cstring>
#include <string>

#include <cuda.h>
//#include "helper_cuda.h"
#include "postscript.h"
#include "marsaglia.h"
#include "IntegrationKernels.h"
#include "RandomVector.h"

#include "json/json.h"

#define CudaErrorCheck() { \
      cudaError_t e = cudaGetLastError(); \
      if (e!=cudaSuccess){\
          printf("Cuda failure %s: %d: %s\n", __FILE__, __LINE__, cudaGetErrorString(e)); \
          exit(0); \
      }            \
    }

float mass;                                        //  M
float repulsion_range,    attraction_range;        //  LL1, LL2
float repulsion_strength, attraction_strength;     //  ST1, ST2

// variables to allow for different stiffnesses
float stiffness1;
float stiffness2; 
float* d_Youngs_mod;
float* youngsModArray; 
bool useDifferentStiffnesses;
float softYoungsMod;
int numberOfSofterCells;
bool duringGrowth;
bool daughtSameStiffness;
float closenessToCenter;
int startAtPop;

bool chooseRandomCellIndices;
float fractionOfSofterCells;

float viscotic_damping, internal_damping;          //  C, DMP
float gamma_visc;
float zOffset; // Offset from Z = 0 for starting positions.
int ranZOffset;
int   Time_steps;
float divVol;
float delta_t;
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

// equilibrium length of springs between fullerene atoms
float R0  = 0.13517879937327418f;

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

float* d_velHtsX;
float* d_velHtsY;
float* d_velHtsZ;

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
bool useRigidSimulationBox; 
float* d_boxMin;

int No_of_threads; // ie number of staring cells
int Side_length;
int ex, ey;


float  *X,  *Y,  *Z;     // host: atom positions

float *d_XP, *d_YP, *d_ZP;     // device: time propagated atom positions
float  *d_X,  *d_Y,  *d_Z;     // device: present atom positions
float *d_XM, *d_YM, *d_ZM;     // device: previous atom positions


float* d_Fx;
float* d_Fy;
float* d_Fz;

// host: minimal bounding box for fullerene
float *bounding_xyz;
float *d_bounding_xyz;   // device:  bounding_xyz

// global minimum and maximum of x and y, preprocessfirst
// global minimum and maximum of x and y, postprocesssecond
float *d_Minx, *d_Maxx, *d_Miny, *d_Maxy, *d_Minz, *d_Maxz;
float *Minx, *Maxx, *Miny, *Maxy, *Minz, *Maxz;

float DL;
int Xdiv, Ydiv, Zdiv;

int *d_NoofNNlist;
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


int main(int argc, char *argv[])
{
  int i;
  int globalrank,step;
  int noofblocks, threadsperblock, prevnoofblocks;
  int Orig_No_of_C180s, newcells;
  int reductionblocks;
  //float PSS;
  float s, theta, phi;
  FILE *outfile;
  FILE *trajfile; // pointer to xyz file
  cudaError_t myError;

  int* dividingCells; //Cells that are about to divide
  int* totalCells; // No. of cells at every Dividing_steps


  int* num_new_cells_per_step;
  int countOffset = 0;

  //int min_no_of_cells = 10;

  printf("CellDiv version 0.9\n");

  if ( argc != 3 )
  {
      printf("Usage: CellDiv no_of_threads inpFile.json\n");
      return(0);
  }

  No_of_threads = atoi(argv[1]);

  char inpFile[256];
  strcpy(inpFile, argv[2]);

  if ( read_json_params(inpFile)          != 0 ) return(-1);

  printf("%d\n", MaxNoofC180s); 

  Side_length   = (int)( sqrt( (double)No_of_threads )+0.5);
  if ( No_of_threads > MaxNoofC180s || Side_length*Side_length != No_of_threads )
  {
      printf("Usage: Celldiv no_of_threads\n");
      printf("       no_of_threads should be a square, n^2, < %d\n", MaxNoofC180s);
      return(0);
  }


  No_of_C180s      = No_of_threads;
  Orig_No_of_C180s = No_of_C180s;
  GPUMemory = 0L;
  CPUMemory = 0L;

  //if ( read_global_params()               != 0 ) return(-1);
  if ( read_fullerene_nn()                != 0 ) return(-1);
  if ( generate_random(Orig_No_of_C180s)  != 0 ) return(-1);
  if ( initialize_C180s(Orig_No_of_C180s) != 0 ) return(-1);
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



  // Allocate initializing memory for didCellDie

  didCellDie = (char *)(calloc(MaxNoofC180s, sizeof(char)));
  CPUMemory += (long)MaxNoofC180s * sizeof(char);

  cudaDeviceProp deviceProp = getDevice();

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
  if ( cudaSuccess != cudaMalloc( (void **)&d_volume ,       MaxNoofC180s*sizeof(float))) return(-1);
  if ( cudaSuccess != cudaMalloc( (void **)&d_cell_div ,     MaxNoofC180s*sizeof(char))) return(-1);
  if ( cudaSuccess != cudaMalloc( (void **)&d_Minx ,         1024*sizeof(float))) return(-1);
  if ( cudaSuccess != cudaMalloc( (void **)&d_Maxx ,         1024*sizeof(float))) return(-1);
  if ( cudaSuccess != cudaMalloc( (void **)&d_Miny ,         1024*sizeof(float))) return(-1);
  if ( cudaSuccess != cudaMalloc( (void **)&d_Maxy ,         1024*sizeof(float))) return(-1);
  if ( cudaSuccess != cudaMalloc( (void **)&d_Minz ,         1024*sizeof(float))) return(-1);
  if ( cudaSuccess != cudaMalloc( (void **)&d_Maxz ,         1024*sizeof(float))) return(-1);
  if ( cudaSuccess != cudaMalloc( (void **)&d_NoofNNlist ,   1024*1024*sizeof(int))) return(-1);
  if ( cudaSuccess != cudaMalloc( (void **)&d_NNlist ,    32*1024*1024*sizeof(int))) return(-1);
  if ( cudaSuccess != cudaMalloc( (void **)&d_C180_56,       92*7*sizeof(int))) return(-1);
  if ( cudaSuccess != cudaMalloc( (void **)&d_ran2 , 10000*sizeof(float))) return(-1);
  if ( cudaSuccess != cudaMalloc( (void **)&d_pressList, MaxNoofC180s*sizeof(float))) return(-1);
  if ( cudaSuccess != cudaMalloc( (void **)&d_velListX, 192*MaxNoofC180s*sizeof(float))) return(-1);
  if ( cudaSuccess != cudaMalloc( (void **)&d_velListY, 192*MaxNoofC180s*sizeof(float))) return(-1);
  if ( cudaSuccess != cudaMalloc( (void **)&d_velListZ, 192*MaxNoofC180s*sizeof(float))) return(-1);
  if ( cudaSuccess != cudaMalloc( (void **)&d_velHtsX, 192*MaxNoofC180s*sizeof(float))) return(-1);
  if ( cudaSuccess != cudaMalloc( (void **)&d_velHtsY, 192*MaxNoofC180s*sizeof(float))) return(-1);
  if ( cudaSuccess != cudaMalloc( (void **)&d_velHtsZ, 192*MaxNoofC180s*sizeof(float))) return(-1);
  if ( cudaSuccess != cudaMalloc( (void **)&d_resetIndices, MaxNoofC180s*sizeof(int))) return(-1);
  if ( cudaSuccess != cudaMalloc( (void **)&d_Fx, 192*MaxNoofC180s*sizeof(float))) return(-1);
  if ( cudaSuccess != cudaMalloc( (void **)&d_Fy, 192*MaxNoofC180s*sizeof(float))) return(-1);
  if ( cudaSuccess != cudaMalloc( (void **)&d_Fz, 192*MaxNoofC180s*sizeof(float))) return(-1);
  if ( cudaSuccess != cudaMalloc( (void **)&d_Youngs_mod, MaxNoofC180s*sizeof(float))) return(-1);
  if ( cudaSuccess != cudaMalloc( (void **)&d_boxMin, 3*sizeof(float))) return(-1); 
  


  bounding_xyz = (float *)calloc(MaxNoofC180s*6, sizeof(float));
  CMx   = (float *)calloc(MaxNoofC180s, sizeof(float));
  CMy   = (float *)calloc(MaxNoofC180s, sizeof(float));
  CMz   = (float *)calloc(MaxNoofC180s, sizeof(float));
  volume= (float *)calloc(MaxNoofC180s, sizeof(float));
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


  cudaMemcpy(d_pressList, pressList, MaxNoofC180s*sizeof(float), cudaMemcpyHostToDevice);

  velListX = (float *)calloc(192*MaxNoofC180s, sizeof(float)); 
  velListY = (float *)calloc(192*MaxNoofC180s, sizeof(float)); 
  velListZ = (float *)calloc(192*MaxNoofC180s, sizeof(float));

  cudaMemcpy(d_velListX, velListX, 192*MaxNoofC180s*sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_velListY, velListY, 192*MaxNoofC180s*sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_velListZ, velListZ, 192*MaxNoofC180s*sizeof(float), cudaMemcpyHostToDevice);

  cudaMemcpy(d_velHtsX, velListX, 192*MaxNoofC180s*sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_velHtsY, velListY, 192*MaxNoofC180s*sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_velHtsZ, velListZ, 192*MaxNoofC180s*sizeof(float), cudaMemcpyHostToDevice);

  cudaMemcpy(d_Fx, velListX, 192*MaxNoofC180s*sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_Fy, velListY, 192*MaxNoofC180s*sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_Fz, velListZ, 192*MaxNoofC180s*sizeof(float), cudaMemcpyHostToDevice);
  
  cudaMemcpy(d_volume, velListZ, MaxNoofC180s*sizeof(float), cudaMemcpyHostToDevice); 

  // Set the Youngs_mod for the cells
  youngsModArray = (float *)calloc(MaxNoofC180s, sizeof(float));
  if (useDifferentStiffnesses){
      
      if (!duringGrowth){
          
          for (int i = 0; i < MaxNoofC180s; i++){
              youngsModArray[i] = stiffness1;
          }
          
      } else {
          
          if (fractionOfSofterCells > 0){
              
              int c = 0;
              for (int i = 0; i < MaxNoofC180s; i++){
                  float ran1[1];
                  ranmar(ran1, 1);
                  if (ran1[0] <= fractionOfSofterCells){
                      youngsModArray[i] = stiffness2;
                      c++; 
                  }
                  else
                      youngsModArray[i] = stiffness1;
                  
              }
              
              float fset = ((float)c)/((float)MaxNoofC180s);
              if ( abs(fset - fractionOfSofterCells) > 1e-1 )
                  printf("WARNING: %.2f %% cells set to softer, %.2f %% requested\n",
                         fset*100, fractionOfSofterCells*100);
              
          } else if (numberOfSofterCells > 0){
              
              if (!chooseRandomCellIndices){
                  printf("ERROR: Cell indices can only be chose randomly during growth\n");
                  return -11;
              }
              
              for (int i = 0; i < numberOfSofterCells; i++){
                  youngsModArray[i] = stiffness2; 
              }

               for (int i = numberOfSofterCells; i < MaxNoofC180s; i++){
                  youngsModArray[i] = stiffness1; 
              }
              
          }
          
      }
  } else if (!useDifferentStiffnesses){
      
      for (int i = 0; i < MaxNoofC180s; i++){
          youngsModArray[i] = stiffness1;
      }
  }

  for (int i = 0; i < MaxNoofC180s; i++){
      printf("cell: %d, stiffness = %f\n", i, youngsModArray[i]);
  }
  
  
  cudaMemcpy(d_Youngs_mod, youngsModArray, MaxNoofC180s*sizeof(float), cudaMemcpyHostToDevice);
  CudaErrorCheck();
        
  // Better way to see how much GPU memory is being used.
  size_t totalGPUMem;
  size_t freeGPUMem;

  if ( cudaSuccess != cudaMemGetInfo ( &freeGPUMem, &totalGPUMem ) ) {
      printf("Couldn't read GPU Memory status\nExiting...");
      exit(1);
  }

  GPUMemory = totalGPUMem - freeGPUMem;

  printf("   Total amount of GPU memory used =    %8.2lf MB\n",GPUMemory/1000000.0);
  printf("   Total amount of CPU memory used =    %8.2lf MB\n",CPUMemory/1000000.0);

  cudaMemcpy(d_C180_nn,   C180_nn,   3*192*sizeof(int),cudaMemcpyHostToDevice);
  cudaMemcpy(d_C180_sign, C180_sign, 180*sizeof(int),cudaMemcpyHostToDevice);
  cudaMemcpy(d_C180_56,   C180_56,   7*92*sizeof(int),cudaMemcpyHostToDevice);

  cudaMemcpy(d_XP, X, 192*MaxNoofC180s*sizeof(float),cudaMemcpyHostToDevice);
  cudaMemcpy(d_YP, Y, 192*MaxNoofC180s*sizeof(float),cudaMemcpyHostToDevice);
  cudaMemcpy(d_ZP, Z, 192*MaxNoofC180s*sizeof(float),cudaMemcpyHostToDevice);
  cudaMemcpy(d_X,  X, 192*MaxNoofC180s*sizeof(float),cudaMemcpyHostToDevice);
  cudaMemcpy(d_Y,  Y, 192*MaxNoofC180s*sizeof(float),cudaMemcpyHostToDevice);
  cudaMemcpy(d_Z,  Z, 192*MaxNoofC180s*sizeof(float),cudaMemcpyHostToDevice);
  cudaMemcpy(d_XM, X, 192*MaxNoofC180s*sizeof(float),cudaMemcpyHostToDevice);
  cudaMemcpy(d_YM, Y, 192*MaxNoofC180s*sizeof(float),cudaMemcpyHostToDevice);
  cudaMemcpy(d_ZM, Z, 192*MaxNoofC180s*sizeof(float),cudaMemcpyHostToDevice);

  cudaMemcpy(d_cell_div, cell_div, MaxNoofC180s*sizeof(char), cudaMemcpyHostToDevice);


  prevnoofblocks  = No_of_C180s;
  noofblocks      = No_of_C180s;
  threadsperblock = 192;
  printf("   no of blocks = %d, threadsperblock = %d, no of threads = %ld\n",
         noofblocks, threadsperblock, ((long) noofblocks)*((long) threadsperblock));

  bounding_boxes<<<No_of_C180s,32>>>(No_of_C180s,d_XP,d_YP,d_ZP,d_X,d_Y,d_Z,d_XM,d_YM,d_ZM,
                                     d_bounding_xyz, d_CMx, d_CMy, d_CMz);


  reductionblocks = (No_of_C180s-1)/1024+1;
  minmaxpre<<<reductionblocks,1024>>>( No_of_C180s, d_bounding_xyz,
                                       d_Minx, d_Maxx, d_Miny, d_Maxy, d_Minz, d_Maxz);
  CudaErrorCheck(); 
  minmaxpost<<<1,1024>>>(reductionblocks, d_Minx, d_Maxx, d_Miny, d_Maxy, d_Minz, d_Maxz);
  CudaErrorCheck(); 
  cudaMemset(d_NoofNNlist, 0, 1024*1024);
  cudaMemcpy(Minx, d_Minx, 6*sizeof(float),cudaMemcpyDeviceToHost);
  //  DL = 3.8f;
  DL = 2.9f;
  //DL = divVol; 
  Xdiv = (int)((Minx[1]-Minx[0])/DL+1);
  Ydiv = (int)((Minx[3]-Minx[2])/DL+1);
  Zdiv = (int)((Minx[5]-Minx[4])/DL+1);
  makeNNlist<<<No_of_C180s/512+1,512>>>( No_of_C180s, d_bounding_xyz, Minx[0], Minx[2], Minx[4],
                                         attraction_range, Xdiv, Ydiv, Zdiv, d_NoofNNlist, d_NNlist, DL);
  CudaErrorCheck(); 
  globalrank = 0;

  // open trajectory file
  trajfile = fopen (trajFileName, "w");
  if ( trajfile == NULL)
  {
      printf("Failed to open %s \n", trajFileName);
      return -1;
  }

  FILE* velFile = fopen("velocity2.xyz", "w"); 

  //OpenBinaryFile("binFile.hist", &bFA, trajWriteInt);

  write_traj(0, trajfile);
 // WriteToBinaryFile(X, Y, Z,
 //                    No_of_C180s, 0, &bFA);

  // Set up walls if needed
  if (useWalls == 1){
      // First we must make sure that the walls surround the
      // starting system.
      CenterOfMass<<<No_of_C180s,256>>>(No_of_C180s,
                                        d_XP, d_YP, d_ZP,
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

  float rGrowth = 0;
  bool growthDone = false;

  boxMin[0] = 0;
  boxMin[1] = 0;
  boxMin[2] = 0;
  
  // Setup simulation box, if needed (non-pbc)
  if (useRigidSimulationBox){
      printf("   Setup rigid (non-PBC) box...\n"); 
      boxLength = ceil(max( (Minx[5]-Minx[4]), max( (Minx[1]-Minx[0]), (Minx[3]-Minx[2]) ) ));
      if (Side_length < 5) boxLength = boxLength * 5; 
      boxMin[0] = floor(Minx[0] - 0.1);
      boxMin[1] = floor(Minx[2] - 0.1);
      boxMin[2] = floor(Minx[4] - 0.1);
      printf("   Done!\n");
      printf("   Simulation box minima:\n   X: %f, Y: %f, Z: %f\n", boxMin[0], boxMin[1], boxMin[2]);
      printf("   Simulation box length = %f\n", boxLength);
  }

  cudaMemcpy(d_boxMin, boxMin, 3*sizeof(float), cudaMemcpyHostToDevice); 
  
  // Simulation loop
  for ( step = 1; step < Time_steps+1 + equiStepCount; step++)
  {
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
      PressureUpdate <<<No_of_C180s/512 + 1, 512>>> (d_pressList, minPressure, maxPressure, rGrowth, No_of_C180s);
      
      if ( (step)%1000 == 0)
      {
          printf("   time %-8d %d cells, rGrowth %f, maxPop %f\n",step,No_of_C180s, rGrowth, maxPop);
      }

      noofblocks      = No_of_C180s;
      if ( prevnoofblocks < noofblocks )
      {
          prevnoofblocks  = noofblocks;
          //        printf("             no of thread blocks = %d, threadsperblock = %d, no of threads = %ld\n",
          //             noofblocks, threadsperblock, ((long) noofblocks)*((long) threadsperblock));
      }

#ifdef FORCE_DEBUG
      printf("time %d  pressure = %f\n", step, Pressure);
#endif
          //printf("\n new step \n"); 
      propagate<<<noofblocks,threadsperblock>>>( No_of_C180s, d_C180_nn, d_C180_sign,
                                                 d_XP, d_YP, d_ZP, d_X,  d_Y,  d_Z, d_XM, d_YM, d_ZM,
                                                 d_CMx, d_CMy, d_CMz,
                                                 R0, d_pressList, d_Youngs_mod , stiffness1, 
                                                 internal_damping, delta_t, d_bounding_xyz,
                                                 attraction_strength, attraction_range,
                                                 repulsion_strength, repulsion_range,
                                                 viscotic_damping, mass,
                                                 Minx[0], Minx[2], Minx[4], Xdiv, Ydiv, Zdiv, d_NoofNNlist, d_NNlist, DL, gamma_visc,
                                                 wall1, wall2,
                                                 threshDist, useWalls,
                                                 d_velListX, d_velListY, d_velListZ,
                                                 useRigidSimulationBox, boxLength, d_boxMin);
      
      CenterOfMass<<<No_of_C180s,256>>>(No_of_C180s,
                                        d_XP, d_YP, d_ZP,
                                        d_CMx, d_CMy, d_CMz);
      if (step <= Time_steps && rGrowth > 0){
        // ------------------------------ Begin Cell Division ------------------------------------------------


        volumes<<<No_of_C180s,192>>>(No_of_C180s, d_C180_56,
                                     d_XP, d_YP, d_ZP,
                                     d_CMx , d_CMy, d_CMz,
                                     d_volume, d_cell_div, divVol);

        count_and_get_div();
        for (int divCell = 0; divCell < num_cell_div; divCell++) {
          globalrank = cell_div_inds[divCell];
          float norm[3];

          if (useDivPlaneBasis)
              GetRandomVectorBasis(norm, divPlaneBasis);
          else
              GetRandomVector(norm); 

          cudaMemcpy( d_ran2, norm, 3*sizeof(float), cudaMemcpyHostToDevice); 
          
          NDIV[globalrank] += 1;

          cell_division<<<1,256>>>(globalrank,
                                   d_XP, d_YP, d_ZP,
                                   d_X, d_Y, d_Z,
                                   d_CMx, d_CMy, d_CMz,
                                   No_of_C180s, d_ran2, repulsion_range);
          resetIndices[divCell] = globalrank;
          resetIndices[divCell + num_cell_div] = No_of_C180s;

          if (daughtSameStiffness){
              youngsModArray[No_of_C180s] = youngsModArray[globalrank];
              cudaMemcpy(d_Youngs_mod+No_of_C180s, youngsModArray+No_of_C180s,
                         sizeof(float), cudaMemcpyHostToDevice);
          }

          ++No_of_C180s;
        }

        cudaMemcpy(d_resetIndices, resetIndices, 2*num_cell_div*sizeof(int),
                   cudaMemcpyHostToDevice); 

        PressureReset <<<(2*num_cell_div)/512 + 1, 512>>> (d_resetIndices, d_pressList, minPressure, 2*num_cell_div); 

        totalFood -= num_cell_div*cellFoodConsDiv;

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
                                         d_XP,d_YP,d_ZP,d_X,d_Y,d_Z,d_XM,d_YM,d_ZM,
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
                          printf("%d, ", cellInd); 
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

      if ( step%trajWriteInt == 0 )
      {
          //printf("   Writing trajectory to traj.xyz...\n");
          cudaMemcpy(X, d_X, 192*No_of_C180s*sizeof(float),cudaMemcpyDeviceToHost);
          cudaMemcpy(Y, d_Y, 192*No_of_C180s*sizeof(float),cudaMemcpyDeviceToHost);
          cudaMemcpy(Z, d_Z, 192*No_of_C180s*sizeof(float),cudaMemcpyDeviceToHost);

          write_traj(step, trajfile);

          cudaMemcpy(velListX, d_velListX, 192*No_of_C180s*sizeof(float),cudaMemcpyDeviceToHost);
          cudaMemcpy(velListY, d_velListY, 192*No_of_C180s*sizeof(float),cudaMemcpyDeviceToHost);
          cudaMemcpy(velListZ, d_velListZ, 192*No_of_C180s*sizeof(float),cudaMemcpyDeviceToHost);
          
          write_vel(step, velFile); 
          // WriteToBinaryFile(X, Y, Z,
          //                   No_of_C180s, step, &bFA);
      }

#if defined(FORCE_DEBUG) || defined(PRINT_VOLUMES)
      volumes<<<No_of_C180s,192>>>(No_of_C180s, d_C180_56,
                                     d_XP, d_YP, d_ZP,
                                     d_CMx , d_CMy, d_CMz,
                                     d_volume, d_cell_div, divVol*100);
      
      cudaMemcpy(volume, d_volume, No_of_C180s*sizeof(float), cudaMemcpyDeviceToHost);
      for (int i = 0; i < No_of_C180s; i++){
          printf ("Cell: %d, volume= %f\n", i, volume[i]);
      }
#endif



      myError = cudaGetLastError();
      if ( cudaSuccess != myError )
      {
          printf( "Error %d: %s!\n",myError,cudaGetErrorString(myError) );return(-1);
      }
  }


  // Write postscript file
  //cudaMemcpy(X, d_X, 192*No_of_C180s*sizeof(float),cudaMemcpyDeviceToHost);
  //cudaMemcpy(Y, d_Y, 192*No_of_C180s*sizeof(float),cudaMemcpyDeviceToHost);
  //cudaMemcpy(Z, d_Z, 192*No_of_C180s*sizeof(float),cudaMemcpyDeviceToHost);
  //PSNET(No_of_C180s*270,Side_length,L1,X,Y,Z,CCI);

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
  fclose(MitIndFile);
  // CloseBinaryFile(&bFA);
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

  ranmar(ran2,Orig_No_of_C180s);

  for ( rank = 0; rank < Orig_No_of_C180s; ++rank )
  {
      ey=rank%Side_length;
      ex=rank/Side_length;

      if (ranZOffset) // Branch prediction should kick in
          zOffset += ran2[rank]-0.5;

      for ( atom = 0 ; atom < 180 ; ++atom)
      {
          X[rank*192+atom] = initx[atom] + L1*ex + 0.5*L1;
          Y[rank*192+atom] = inity[atom] + L1*ey + 0.5*L1;
          Z[rank*192+atom] = initz[atom] + zOffset;
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

  time_t current_time;
  time(&current_time);
  seed_ij = (int)current_time;
  localtime(&current_time);
  seed_kl = (int)current_time;
  ij = seed_ij%31328;
  kl = seed_kl%30081;
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
        stiffness1 = coreParams["Youngs_mod"].asFloat();
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

        maxPressure = coreParams["maxPressure"].asFloat();
        minPressure = coreParams["minPressure"].asFloat();
        gamma_visc = coreParams["gamma_visc"].asFloat();
        rMax = coreParams["growth_rate"].asFloat();     

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
        stiffness2 = stiffnessParams["softYoungsMod"].asFloat();
        numberOfSofterCells = stiffnessParams["numberOfSofterCells"].asInt();
        duringGrowth = stiffnessParams["duringGrowth"].asBool(); 
        closenessToCenter = stiffnessParams["closenessToCenter"].asFloat();
        startAtPop = stiffnessParams["startAtPop"].asInt();
        fractionOfSofterCells = stiffnessParams["fractionOfSofterCells"].asFloat();
        chooseRandomCellIndices = stiffnessParams["chooseRandomCellIndices"].asBool();
        daughtSameStiffness = stiffnessParams["daughtSameStiffness"].asBool(); 
    }

    Json::Value boxParams = inpRoot.get("boxParams", Json::nullValue);

    if (boxParams == Json::nullValue){
        printf("ERROR: Cannot load box parameters\n");
        return -1;
    } else{
        useRigidSimulationBox = boxParams["useRigidSimulationBox"].asBool();
    }
    
    
    if (ranZOffset == 0)
        zOffset = 0.0;


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
    printf("      useRigidSimulationBox = %d\n", useRigidSimulationBox);
    
    



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



__global__ void propagate( int No_of_C180s, int d_C180_nn[], int d_C180_sign[],
                           float d_XP[], float d_YP[], float d_ZP[],
                           float d_X[],  float d_Y[],  float d_Z[],
                           float d_XM[], float d_YM[], float d_ZM[],
                           float *d_CMx, float *d_CMy, float *d_CMz,
                           float R0, float* d_pressList, float* d_Youngs_mod , float bondingYoungsMod, 
                           float internal_damping, float delta_t,
                           float d_bounding_xyz[],
                           float attraction_strength, float attraction_range,
                           float repulsion_strength, float repulsion_range,
                           float viscotic_damping, float mass,
                           float Minx, float Miny,  float Minz, int Xdiv, int Ydiv, int Zdiv,
                           int *d_NoofNNlist, int *d_NNlist, float DL, float gamma_visc,
                           float wall1, float wall2,
                           float threshDist, bool useWalls, 
                           float* d_velListX, float* d_velListY, float* d_velListZ,
                           bool useRigidSimulationBox, float boxLength, float* d_boxMin)
{
#ifdef FORCE_DEBUG
        __shared__ float FX_sum;
        __shared__ float FY_sum;
        __shared__ float FZ_sum;
        if (threadIdx.x == 0){
            FX_sum = 0;
            FY_sum = 0;
            FZ_sum = 0;
        }

        __syncthreads();

#endif

    int rank, atom, nn_rank, nn_atom;
    int N1, N2, N3;
    int NooflocalNN;
    int localNNs[10];
    float deltaX, deltaY, deltaZ;
    float A1, A2, A3;
    float B1, B2, B3;
    float TX, TY, TZ;
    float NORM, R;
    float NX, NY, NZ;
    //float setPress;

    rank = blockIdx.x;
    atom = threadIdx.x;
    float Pressure = d_pressList[rank]; 
    int cellOffset = rank*192;
    int atomInd = cellOffset + atom;
    float Youngs_mod;
    
    if ( rank < No_of_C180s && atom < 180 )
    {
        if (isnan(d_X[rank*192+atom]) ||
            isnan(d_Y[rank*192+atom]) || 
            isnan(d_Z[rank*192+atom])){
            printf("OH SHIT: we have a nan\n");
            printf("Particle index: %d\n", atom);
            printf("Crash now :(\n"); 
            asm("trap;"); 
        }

        Youngs_mod = d_Youngs_mod[rank];
        
        N1 = d_C180_nn[  0+atom];
        N2 = d_C180_nn[192+atom];
        N3 = d_C180_nn[384+atom];

        A1 = d_X[rank*192+N2]-d_X[rank*192+N1];
        A2 = d_Y[rank*192+N2]-d_Y[rank*192+N1];
        A3 = d_Z[rank*192+N2]-d_Z[rank*192+N1];

        B1 = d_X[rank*192+N3]-d_X[rank*192+N1];
        B2 = d_Y[rank*192+N3]-d_Y[rank*192+N1];
        B3 = d_Z[rank*192+N3]-d_Z[rank*192+N1];

        TX = A2*B3-A3*B2;
        TY = A3*B1-A1*B3;
        TZ = A1*B2-A2*B1;

        NORM = sqrt(TX*TX+TY*TY+TZ*TZ);

        NX = d_C180_sign[atom]*TX/NORM;
        NY = d_C180_sign[atom]*TY/NORM;
        NZ = d_C180_sign[atom]*TZ/NORM;

        float X = d_X[rank*192+atom];
        float Y = d_Y[rank*192+atom];
        float Z = d_Z[rank*192+atom];

        float FX = 0.0f;
        float FY = 0.0f;
        float FZ = 0.0f;

        int nnAtomInd;
        
        
        float velX = d_velListX[atomInd];
        float velY = d_velListY[atomInd];
        float velZ = d_velListZ[atomInd];


        //  Spring Force calculation within cell
        //  go through three nearest neighbors

        float damp_const = internal_damping/delta_t;

        for ( int i = 0; i < 3 ; ++i ) // Better to open this loop
        {
            N1 = d_C180_nn[i*192+atom];

            deltaX = d_X[rank*192+N1]-d_X[rank*192+atom];
            deltaY = d_Y[rank*192+N1]-d_Y[rank*192+atom];
            deltaZ = d_Z[rank*192+N1]-d_Z[rank*192+atom];

            R  = sqrt(deltaX*deltaX+deltaY*deltaY+deltaZ*deltaZ);

            // spring forces
            FX += +Youngs_mod*(R-R0)/R0*deltaX/R;
            FY += +Youngs_mod*(R-R0)/R0*deltaY/R;
            FZ += +Youngs_mod*(R-R0)/R0*deltaZ/R;


            // pressure forces
            FX += Pressure*NX;
            FY += Pressure*NY;
            FZ += Pressure*NZ;

            // internal damping
            // FX += -damp_const*(-deltaX-(d_XM[rank*192+atom]-d_XM[rank*192+N1]));
            // FY += -damp_const*(-deltaY-(d_YM[rank*192+atom]-d_YM[rank*192+N1]));
            // FZ += -damp_const*(-deltaZ-(d_ZM[rank*192+atom]-d_ZM[rank*192+N1]));

            FX += -internal_damping*(d_velListX[atomInd] - d_velListX[rank*192+N1]);
            FY += -internal_damping*(d_velListY[atomInd] - d_velListY[rank*192+N1]);
            FZ += -internal_damping*(d_velListZ[atomInd] - d_velListZ[rank*192+N1]);

        }

#ifdef FORCE_DEBUG

        atomicAdd(&FX_sum, FX);
        __syncthreads();
        atomicAdd(&FY_sum, FY);
        __syncthreads();
        atomicAdd(&FZ_sum, FZ);
        __syncthreads();
        if (threadIdx.x == 0){
            printf("Spring, pressure, internal\n");
            printf("Fx = %f, Fy = %f, Fz = %f\n", FX_sum, FY_sum, FZ_sum);
        }

#endif


        NooflocalNN = 0;

        int startx = (int)((X -Minx)/DL);
        if ( startx < 0 ) startx = 0;
        if ( startx >= Xdiv ) startx = Xdiv-1;

        int starty = (int)((Y - Miny)/DL);
        if ( starty < 0 ) starty = 0;
        if ( starty >= Ydiv ) starty = Ydiv-1;

        int startz = (int)((Z - Minz)/DL);
        if ( startz < 0 ) startz = 0;
        if ( startz >= Zdiv ) startz = Zdiv-1;

        int index = startz*Xdiv*Ydiv + starty*Xdiv + startx;

        // interfullerene attraction and repulsion
        for ( int nn_rank1 = 1 ; nn_rank1 <= d_NoofNNlist[index] ; ++nn_rank1 )
        {
            nn_rank = d_NNlist[32*index+nn_rank1-1];
            if ( nn_rank == rank ) continue;

            deltaX  = (X-d_bounding_xyz[nn_rank*6+1]>0.0f)*(X-d_bounding_xyz[nn_rank*6+1]);
            deltaX += (d_bounding_xyz[nn_rank*6+0]-X>0.0f)*(d_bounding_xyz[nn_rank*6+0]-X);

            deltaY  = (Y-d_bounding_xyz[nn_rank*6+3]>0.0f)*(Y-d_bounding_xyz[nn_rank*6+3]);
            deltaY += (d_bounding_xyz[nn_rank*6+2]-Y>0.0f)*(d_bounding_xyz[nn_rank*6+2]-Y);

            deltaZ  = (Z-d_bounding_xyz[nn_rank*6+5]>0.0f)*(Z-d_bounding_xyz[nn_rank*6+5]);
            deltaZ += (d_bounding_xyz[nn_rank*6+4]-Z>0.0f)*(d_bounding_xyz[nn_rank*6+4]-Z);

            if ( deltaX*deltaX + deltaY*deltaY + deltaZ*deltaZ > attraction_range*attraction_range )
                continue;

            ++NooflocalNN;

            if ( NooflocalNN > 10 ){
                printf("Recoverable error: NooflocalNN = %d, should be < 8\n",NooflocalNN);
                continue;
            }

            localNNs[NooflocalNN-1] = nn_rank;
        }

        for ( int i = 0; i < NooflocalNN; ++i )
        {
            nn_rank =localNNs[i];
            nnAtomInd = nn_rank*192;

            for ( nn_atom = 0; nn_atom < 180 ; ++nn_atom )
            {
                nnAtomInd += nn_atom;

                deltaX = d_X[rank*192+atom]-d_X[nn_rank*192+nn_atom];
                deltaY = d_Y[rank*192+atom]-d_Y[nn_rank*192+nn_atom];
                deltaZ = d_Z[rank*192+atom]-d_Z[nn_rank*192+nn_atom];

                R = deltaX*deltaX+deltaY*deltaY+deltaZ*deltaZ;

                if ( R >= attraction_range*attraction_range )
                    continue;

                R = sqrt(R);

                if ( R < attraction_range )
                {
                    FX += -attraction_strength*Youngs_mod*(attraction_range-R)/R*deltaX;
                    FY += -attraction_strength*Youngs_mod*(attraction_range-R)/R*deltaY;
                    FZ += -attraction_strength*Youngs_mod*(attraction_range-R)/R*deltaZ;

                    // hinder rearrangements

                    // First calculate relative velocity
                    float v_ijx = d_velListX[atomInd] - d_velListX[nn_rank*192+nn_atom];
                    float v_ijy = d_velListY[atomInd] - d_velListY[nn_rank*192+nn_atom];
                    float v_ijz = d_velListZ[atomInd] - d_velListZ[nn_rank*192+nn_atom];

                    // Dot product between relative and normal to surface
                    float vijDotn = v_ijx*NX + v_ijy*NY + v_ijz*NZ; 

                    // Tangential component
                    float vTauX = v_ijx - vijDotn*NX;
                    float vTauY = v_ijy - vijDotn*NY;
                    float vTauZ = v_ijz - vijDotn*NZ; 

                    FX -= viscotic_damping*vTauX;
                    FY -= viscotic_damping*vTauY;
                    FZ -= viscotic_damping*vTauZ;
                }
                if ( R < repulsion_range )
                {
                    if (R < (repulsion_range-0.01)) R = repulsion_range-0.01; 
                    FX += +repulsion_strength*Youngs_mod*(repulsion_range-R)/R*deltaX;
                    FY += +repulsion_strength*Youngs_mod*(repulsion_range-R)/R*deltaY;
                    FZ += +repulsion_strength*Youngs_mod*(repulsion_range-R)/R*deltaZ;

                    // if ( deltaX*(d_CMx[rank]-d_CMx[nn_rank])  +
                    //      deltaY*(d_CMy[rank]-d_CMy[nn_rank])  +
                    //      deltaZ*(d_CMz[rank]-d_CMz[nn_rank]) < 0.0f )
                    // {
                    //     //printf("fullerene %d inside %d?\n",rank, nn_rank);
                    // }
                }

            }

        }

#ifdef FORCE_DEBUG

        if (threadIdx.x == 0){
            FX_sum = 0;
            FY_sum = 0;
            FZ_sum = 0;
        }
        __syncthreads();

        atomicAdd(&FX_sum, FX);
        __syncthreads();
        atomicAdd(&FY_sum, FY);
        __syncthreads();
        atomicAdd(&FZ_sum, FZ);
        __syncthreads();
        if (threadIdx.x == 0){
            printf("neighbours\n");
            printf("Fx = %f, Fy = %f, Fz = %f\n", FX_sum, FY_sum, FZ_sum);
        }



#endif

        // add friction

        FX += -1 * gamma_visc * velX;
        FY += -1 * gamma_visc * velY;
        FZ += -1 * gamma_visc * velZ;

        if (useWalls == 1){
            // Wall repulsion

            float wallDist = d_Z[rank*192+atom] - wall1;
            
            if (abs(wallDist) <= threshDist){
            //repulse
                if (wallDist * FZ < 0) FZ = -FZ;
            }

            wallDist = d_Z[rank*192+atom] - wall2;
            if (abs(wallDist) <= threshDist){
                
                if (wallDist * FZ < 0) FZ = -FZ;

            }

            __syncthreads();
        }

        // add forces from simulation box if needed:

        if (useRigidSimulationBox){
            float gap1, gap2;
            
            // X
            
            gap1 = d_X[atomInd] - d_boxMin[0];
            gap2 = d_boxMin[0] + boxLength - d_X[atomInd];
            
            if (abs(gap1) < threshDist && gap1*FX < 0) FX = -FX;
            if (abs(gap2) < threshDist && gap2*FX < 0) FX = -FX;

            // Y

            gap1 = d_Y[atomInd] - d_boxMin[1];
            gap2 = d_boxMin[1] + boxLength - d_Y[atomInd];

            if (abs(gap1) < threshDist && gap1*FY < 0) FY = -FY;
            if (abs(gap2) < threshDist && gap2*FY < 0) FY = -FY;

            // Z
            gap1 = d_Z[atomInd] - d_boxMin[2];
            gap2 = d_boxMin[2] + boxLength - d_Z[atomInd];

            if (abs(gap1) < threshDist && gap1*FZ < 0) FZ = -FZ;
            if (abs(gap2) < threshDist && gap2*FZ < 0) FZ = -FZ;
        }

        // time propagation
        
        d_XP[rank*192+atom] =
            1.0/(1.0+delta_t/(2*mass))*
            ((delta_t*delta_t/mass)*FX+2*d_X[rank*192+atom]+(delta_t/(2*mass)-1.0)*d_XM[rank*192+atom]);
        d_YP[rank*192+atom] =
            1.0/(1.0+delta_t/(2*mass))*
            ((delta_t*delta_t/mass)*FY+2*d_Y[rank*192+atom]+(delta_t/(2*mass)-1.0)*d_YM[rank*192+atom]);
        d_ZP[rank*192+atom] =
            1.0/(1.0+delta_t/(2*mass))*
            ((delta_t*delta_t/mass)*FZ+2*d_Z[rank*192+atom]+(delta_t/(2*mass)-1.0)*d_ZM[rank*192+atom]);

        d_velListX[rank*192+atom] = (d_XP[atomInd] - d_XM[atomInd])/(2*delta_t); 
        d_velListY[rank*192+atom] = (d_YP[atomInd] - d_YM[atomInd])/(2*delta_t); 
        d_velListZ[rank*192+atom] = (d_ZP[atomInd] - d_ZM[atomInd])/(2*delta_t);

    }
}

void write_traj(int t_step, FILE* trajfile)
{

  fprintf(trajfile, "%d\n", No_of_C180s * 192);
  fprintf(trajfile, "Step: %d\n", t_step);
  
  if (useDifferentStiffnesses){
      
        for (int p = 0; p < No_of_C180s*192; p++)
        {
            int cellInd = p/192; 
            if (youngsModArray[cellInd] == stiffness1)
                fprintf(trajfile, "%.7f,  %.7f,  %.7f,  H\n", X[p], Y[p], Z[p]);
            else if(youngsModArray[cellInd] == stiffness2)
                fprintf(trajfile, "%.7f,  %.7f,  %.7f,  C\n", X[p], Y[p], Z[p]);

        }
        
  } else {
      
      for (int p = 0; p < No_of_C180s*192; p++)
      {
          fprintf(trajfile, "%.7f,  %.7f,  %.7f\n", X[p], Y[p], Z[p]);
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

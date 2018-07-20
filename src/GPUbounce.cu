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
// #include <thrust/host_vector.h>
// #include <thrust/device_vector.h>
// #include <thrust/fill.h>
// #include <thrust/sort.h>
// #include <thrust/extrema.h>
//#include "helper_cuda.h"
#include "postscript.h"
#include "marsaglia.h"
//#include "IntegrationKernels.h"
#include "RandomVector.h"
#include "VectorFunctions.hpp"
#include "AdaptiveTimeKernels.cuh"
#include "SimParams.cuh"
#include "TrajWriter.cuh"
#include "State.cuh"
#include "SimState.cuh"
#include "globals.cuh"
#include "json/json.h"


void CudaFailure();

#define CudaErrorCheck() { \
        cudaDeviceSynchronize();\
        cudaError_t e = cudaPeekAtLastError();        \
        if (e!=cudaSuccess){                                            \
            printf("Cuda failure in %s, line %d, code %d %s\n", __FILE__, __LINE__, e, cudaGetErrorString(e)); \
            exit(0); \
        }\
    }


size_t base_n::used_host_mem = 0;

int main(int argc, char *argv[])
{
  cudaError_t myError;

  if ( argc !=4 )
  {
      printf("Usage: CellDiv no_of_threads inpFile.json gpuid\n");
      return(0);
  }

#warning "Hard coded input file"  
  std::string inputFile("inp.json");
  sim_params_struct sim_params;
  ReadSimParams(sim_params, "inp.json");

  sim_params.core_params.init_num_cells = atoi(argv[1]);
  
  SimState simState(atoi(argv[1]), sim_params);

  std::cout << simState.pos.x.h[0] << std::endl;
  
  char inpFile[256];
  strcpy(inpFile, argv[2]);

  TrajWriter TW(sim_params);

  if ( InitializeRandom(sim_params.rand_params.rand_seed)  != 0 ) return(-1);
  if ( initialize_C180s(simState, sim_params) != 0 ) return(-1);
  if ( read_fullerene_nn(simState)                != 0 ) return(-1);

  /* PM
     Allocate memory for the dividingCells array that will be used to
     calculate the mitotic index.
  */

#warning mitotic index related code is not implemented
  // dividingCells = (int *)calloc((Time_steps/newCellCountInt), sizeof(int));
  // totalCells = (int *)calloc((Time_steps/newCellCountInt), sizeof(int));
  // num_new_cells_per_step = (int *)calloc(Time_steps, sizeof(int));


  cudaDeviceProp deviceProp = getDevice();
  if (cudaSuccess != cudaSetDevice(atoi(argv[3]))){
      printf("Could not set to divice %d\n", 2);
      return -1;
  }
  
  simState.bondStiffness.Fill(sim_params.core_params.bond_stiff);  

  if (sim_params.stiff_params.use_diff_stiff){
      std::cout << "Stiffness customization not supported " << std::endl;
      exit(1);
      // printf("Making some cells softer... \n");
      // if (closenessToCenter > 0.f && closenessToCenter < 1.f){
      //     printf("Only making cells within %f of max radius softer\n", closenessToCenter);
      //     CenterOfMass<<<No_of_C180s,256>>>(No_of_C180s,
      //                                       d_XP, d_YP, d_ZP,
      //                                       d_CMx, d_CMy, d_CMz);
          
      //     cudaMemcpy(CMx, d_CMx, No_of_C180s*sizeof(float), cudaMemcpyDeviceToHost);
      //     cudaMemcpy(CMy, d_CMy, No_of_C180s*sizeof(float), cudaMemcpyDeviceToHost);
      //     cudaMemcpy(CMz, d_CMz, No_of_C180s*sizeof(float), cudaMemcpyDeviceToHost);

      //     float3 sysCM = make_float3(0.f, 0.f, 0.f);

      //     for(int i =0; i < No_of_C180s; ++i){
      //         sysCM = sysCM + make_float3(CMx[i], CMy[i], CMz[i]);
      //     }

      //     sysCM = sysCM/No_of_C180s; 
          
      //     printf("COM = (%f, %f, %f)\n", sysCM.x, sysCM.y, sysCM.z);

      //     float rMax = 0;
      //     float mags[No_of_C180s];
          
      //     for (int i =0; i < No_of_C180s; ++i){
      //         float3 pos = make_float3(CMx[i], CMy[i], CMz[i]) - sysCM;
      //         mags[i] = mag(pos);
      //         rMax = max(rMax, mags[i]);
      //     }
      //     int c = 0; 
      //     for (int i = 0; i < No_of_C180s; ++i){
      //         if (mags[i]/rMax <= closenessToCenter){
      //             youngsModArray[i] = stiffness2;
      //             ++c; 
      //         }
      //     }

      //     printf("Made %d cells softer\n", c);
          
      // } else {

      //     printf("Choosing softer cells randomly\n");
      //     int c = numberOfSofterCells;
      //     if (fractionOfSofterCells > 0.f && fractionOfSofterCells < 1.f){
      //         c = round(fractionOfSofterCells*(float)No_of_C180s);
      //     }

      //     if (c > No_of_C180s){
      //         printf("ERROR: Too many softer cells requested\n");
      //         return 12516;
      //     }

      //     if (useRigidSimulationBox){
      //         for (int i =0; i < c; ++i){
      //             youngsModArray[i] = stiffness2; 
      //         }
      //     } else {
      //         float rands[1];
      //         for (int i =0; i < c; ++i){
      //             ranmar(rands, 1);
      //             int ind = round(rands[0]*No_of_C180s);
      //             youngsModArray[ind] = stiffness2; 
      //         }
      //     }
      //     printf("Made %d cells softer\n", c);
      // }
  }

  simState.bondStiffness.CopyToDevice();

  // initialize device rng
    if (sim_params.rand_params.add_rands == true){
        std::cout << "Random number addition not supported for now" << std::endl;
        exit(1);
      // curandGenerator_t gen;
      
      // if (cudaMalloc((void **)&d_rngStates, sizeof(curandState)*192*MaxNoofC180s) != cudaSuccess){
      //     fprintf(stderr, "ERROR: Failed to allocate rng state memory in %s, at %d\n", __FILE__, __LINE__);
      //     return 1;
      // }
          

      // if (cudaMalloc((void **)&d_seeds, sizeof(unsigned int)*192*MaxNoofC180s) != cudaSuccess){
      //     fprintf(stderr, "ERROR: Failed to allocate rng seeds in %s, at %d\n", __FILE__, __LINE__);
      //     return 1;
      // }
      
      // curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_MT19937);
      // CudaErrorCheck();

      // curandSetPseudoRandomGeneratorSeed(gen, time(NULL));
      // CudaErrorCheck();

      // if (rand_seed > 0){
      //     curandSetPseudoRandomGeneratorSeed(gen, rand_seed);
      //     CudaErrorCheck();
      // }

      // curandGenerate(gen, d_seeds, MaxNoofC180s*192);
      // CudaErrorCheck();
  
      // DeviceRandInit<<<(192*MaxNoofC180s)/1024 + 1, 1024>>>(d_rngStates, d_seeds, 192*MaxNoofC180s);
      // CudaErrorCheck();
  }


  int noofblocks      = simState.no_of_cells;
  int threadsperblock = 192;
  printf("   no of blocks = %d, threadsperblock = %d, no of threads = %ld\n",
         noofblocks, threadsperblock, ((long) noofblocks)*((long) threadsperblock));

  CenterOfMass<<<simState.no_of_cells, 256>>>(simState.devPtrs);
  CudaErrorCheck();                                          
  
  bounding_boxes<<<simState.no_of_cells, 32>>>(simState.devPtrs);
                                     
  CudaErrorCheck(); 

  size_t reductionblocks = (simState.no_of_cells-1)/1024+1;
  minmaxpre<<<reductionblocks,1024>>>(simState.devPtrs);
  CudaErrorCheck(); 
  minmaxpost<<<1,1024>>>(simState.devPtrs);
  CudaErrorCheck();
  
  makeNNlist<<<(simState.no_of_cells-1)/1024 + 1,1024>>>(simState.devPtrs, sim_params);
  CudaErrorCheck(); 

  if (sim_params.core_params.correct_com == true){
      CenterOfMass<<<simState.no_of_cells,256>>>(simState.devPtrs);
      simState.cellCOMs.x.CopyToHost();
      simState.cellCOMs.y.CopyToHost();
      simState.cellCOMs.z.CopyToHost();

      real3 sysCM = make_real3(0, 0, 0);

      for (size_t i = 0; i < simState.no_of_cells; ++i){
          auto X = simState.cellCOMs.x.h; 
          auto Y = simState.cellCOMs.z.h; 
          auto Z = simState.cellCOMs.y.h;          
          sysCM = sysCM + make_real3(X[i],
                                     Y[i],
                                     Z[i]);
      }

      sysCM = sysCM/(simState.no_of_cells);
      CorrectCoMMotion<<<(simState.no_of_cells*192)/1024 + 1, 1024>>>(simState.devPtrs, sysCM);
      CudaErrorCheck(); 
  }
  
  

  // Initialize pressures

  simState.pressures.Fill(sim_params.core_params.min_pressure);
  
  if (sim_params.angle_params.angle_pot == true){
      // Code to initialize equillibrium angles
      real3 p, ni, nj, nk;
      R3Nptrs pos = simState.pos.hostPtrs;
      auto C180_nn = simState.C180_nn.h;
      
      for (int n = 0; n<180; n++){
          p = make_real3(pos.x[n], pos.y[n], pos.z[n]); 

          ni = make_real3(pos.x[C180_nn[0*192 + n]], pos.y[C180_nn[0*192 + n]], 
                           pos.z[C180_nn[0*192 + n]]); 
          
          nj = make_real3(pos.x[C180_nn[1*192 + n]], pos.y[C180_nn[1*192 + n]], 
                           pos.z[C180_nn[1*192 + n]]);
          
          nk = make_real3(pos.x[C180_nn[2*192 + n]], pos.y[C180_nn[2*192 + n]],
                           pos.z[C180_nn[2*192 + n]]);

          ni = ni-p;
          nj = nj-p;
          nk = nk-p; 

          simState.theta0.h[n].aij = acos(dot(ni, nj)/(mag(ni)*mag(nj)));
          simState.theta0.h[n].ajk = acos(dot(nj, nk)/(mag(nj)*mag(nk)));
          simState.theta0.h[n].aik = acos(dot(ni, nk)/(mag(ni)*mag(nk)));

      }
      
      simState.theta0.CopyToDevice();
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

  float r_CM_o = pow((3.0/4.0) * (1/3.14159) * sim_params.core_params.division_vol*2.0, 1.0/3);


  std::cout << "Mirrored memory allocated =    " << base_n::used_host_mem/(1024*1024) << "MB" << std::endl;
  
  CalculateConForce<<<simState.no_of_cells,threadsperblock>>>(simState.devPtrs, sim_params); 
  CudaErrorCheck();

  CalculateDisForce<<<simState.no_of_cells, threadsperblock>>>(simState.devPtrs, sim_params);
  CudaErrorCheck();


  TW.WriteState(simState);

  // Simulation loop
  for (long int step = 1; step < sim_params.core_params.div_time_steps + sim_params.core_params.non_div_time_steps; ++step)
  {
      Integrate<<<simState.no_of_cells, threadsperblock>>>(simState.devPtrs, sim_params);
      CudaErrorCheck();

      ForwardTime<<<simState.no_of_cells, threadsperblock>>>(simState.devPtrs);
      CudaErrorCheck();

      PressureUpdate <<<simState.no_of_cells/1024 + 1, 1024>>> (simState.devPtrs, sim_params, step);
      CudaErrorCheck(); 
      
      if ( (step)%1000 == 0)
      {
          printf("   time %-8d %d cells, rGrowth %f, maxPop %f\n", step, simState.no_of_cells);
      }

      CalculateConForce<<<simState.no_of_cells,threadsperblock>>>(simState.devPtrs, sim_params);
      CudaErrorCheck();


      CalculateDisForce<<<simState.no_of_cells, threadsperblock>>>(simState.devPtrs, sim_params);
      CudaErrorCheck();

      // Calculate random Force here...
      // Placeholder
      
      VelocityUpdateA<<<simState.no_of_cells, threadsperblock>>>(simState.devPtrs, sim_params);
      CudaErrorCheck();


      // Dissipative velocity update part...
      for (int s = 0; s < 1; ++s){ // may be looped over more later.
          VelocityUpdateB<<<simState.no_of_cells, threadsperblock>>>(simState.devPtrs, sim_params);
          CudaErrorCheck();

          CalculateDisForce<<<simState.no_of_cells, threadsperblock>>>(simState.devPtrs, sim_params);
      CudaErrorCheck();

      // this loop can be looped until convergence, but that shouldn't
      // be necessary most of the time...
      }

      CenterOfMass<<<simState.no_of_cells,256>>>(simState.devPtrs);
      CudaErrorCheck();
      if (step <= sim_params.core_params.div_time_steps){
        // ------------------------------ Begin Cell Division ------------------------------------------------

          volumes<<<simState.no_of_cells,192>>>(simState.devPtrs, sim_params);
          CudaErrorCheck();

          simState.cellShouldDiv.CopyToHost();

          std::vector<int> divInds;
          
          for (long int cellInd = 0; cellInd < simState.no_of_cells; cellInd++){
              auto csd = simState.cellShouldDiv.h;
              if (csd[cellInd] == 1){
                  divInds.push_back(cellInd);
              }
          }

          simState.cellShouldDiv.Fill(0, simState.no_of_cells);
          std::vector<int> newCellInds;
          for (long int i = 0; i < divInds.size(); ++i){
              long int pCellInd = divInds[i];
              long int dCellInd = simState.no_of_cells;
              if (dCellInd + 1 > sim_params.core_params.max_no_of_cells){
                  printf("ERROR: Population is %d, only allocated enough memory for %d\n",
                         simState.no_of_cells, sim_params.core_params.max_no_of_cells);
                  printf("ERROR: Fatal error, crashing...\n");
                  return -69;
              }
                  
              real3 norm;
              if (sim_params.div_params.use_div_plane_basis == true){
                  norm = GetRandomVectorBasis(sim_params.div_params.div_plane_basis);
              } else {
                  norm = GetRandomVector();
              }

#ifdef TURNOFF_RAN
              norm = make_real3(0, 1, 0);              
#endif

              cell_division<<<1, 192>>>(pCellInd, dCellInd, simState.devPtrs, sim_params, norm);
              CudaErrorCheck();
              simState.numDivisions[pCellInd] += 1;
              newCellInds.push_back(simState.no_of_cells);
              simState.IncrementNumCellsBy(1);
          }
          if (divInds.size() > 0){
              simState.resetIndices.ReadIn(divInds.data(), divInds.size());
              simState.resetIndices.ReadIn(newCellInds.start(), newCellinds.size(), 0, divInds.size());
              
              CudaErrorCheck(); 

              PressureReset <<<(2*divInds.size())/512 + 1, 512>>> (simState.devPtrs, sim_params); 
              CudaErrorCheck();
              simState.no_new_cells = 0;
          }

        // --------------------------------------- End Cell Division -----------
      }
      
      bounding_boxes<<<simState.no_of_cells,32>>>(simState.devPtrs);

      CudaErrorCheck();
          
      long int reductionblocks = (simState.no_of_cells-1)/1024+1;
      minmaxpre<<<reductionblocks,1024>>>(simState.devPtrs);

      CudaErrorCheck(); 

      minmaxpost<<<1,1024>>>(simState.devPtrs);
      
      CudaErrorCheck();

      simState.numOfNNList.Fill(0);

      makeNNlist<<<simState.no_of_cells/512+1,512>>>(simState.devPtrs, sim_params);

      CudaErrorCheck();

      if (step > sim_params.core_params.div_time_steps && !simState.growthDone){
          printf("Cell growth halted.\nProceeding with MD simulation without growth...\n");
          simState.growthDone = true;
      }

      if (sim_params.core_params.correct_com == true){
          simState.cellCOMs.CopyToHost();
          CudaErrorCheck();

          real3 sysCOM = make_real3(0,0,0);

          for (long int cellInd = 0; cellInd < simState.no_of_cells; ++cellInd){
              sysCOM = sysCOM + make_real3(simState.cellCOMs.x.h[cellInd],
                                           simState.cellCOMs.y.h[cellInd],
                                           simState.cellCOMs.z.h[cellInd])/simState.no_of_cells;
          }
          
          CorrectCoMMotion<<<(simState.no_of_cells*192)/1024 + 1, 1024>>>
              (simState.devPtrs, sysCOM);
          CudaErrorCheck(); 
      }

      if ( step%(sim_params.core_params.traj_write_int) == 0 ){

          simState.pos.CopyToHost();
          simState.vel.CopyToHost();
          simState.totForce.CopyToHost();
          simState.vol.CopyToHost();
          
          TW.WriteState(simState);
      }

      myError = cudaGetLastError();
      if ( cudaSuccess != myError )
      {
          printf( "Error %d: %s!\n",myError,cudaGetErrorString(myError) );return(-1);
      }

  }
  //cudaDeviceReset();
}



int initialize_C180s(SimState& simState, const sim_params_struct& sim_params)
{
  int atom;
  thrust::host_vector<real> X(180, 0);
  thrust::host_vector<real> Y(180, 0);
  thrust::host_vector<real> Z(180, 0);
  FILE *infil;

  printf("      Initializing positions for %d fullerenes...\n", simState.no_of_cells);

  infil = fopen("C180","r");
  if ( infil == NULL ) {printf("Unable to open file C180\n");return(-1);}
  
  for (atom = 0 ; atom < 180 ; ++atom)
  {
      if ( fscanf(infil,"%f %f %f",&X[atom], &Y[atom], &Z[atom]) != 3 )
      {
          printf("   Unable to read file C180 on line %d\n",atom+1);
          fclose(infil);
          return(-1);
      }
  }
  fclose(infil);

  // first correct for the cells com

  real3 initCOM = make_real3(0,0,0);
      
  for (int i =0; i < 180; ++i){
      initCOM = initCOM + make_real3(X[i], Y[i], Z[i])/180; 
  }

  for (int i =0; i< 180; ++i){
      X[i] -= initCOM.x;
      Y[i] -= initCOM.y;
      Z[i] -= initCOM.z;
  }

  real rCheck = powf(0.75*(1.f/3.14159)*0.786, 1.f/3.f); // this code is magical
  rCheck *= 1.2;
  
  printf("Check radius = %f\n", rCheck);

  real vol = 0;
  int k = 0;
  
  real3 boxMax = sim_params.box_params.box_max;
  
  vol = boxMax.x*boxMax.y*boxMax.z;
  k = floor(vol/0.786);
      
  if (k < simState.no_of_cells){
      fprintf(stderr, "ERROR: Simulation box is too small\n");
      fprintf(stderr, "       Big enough for %d\n", k);
      return 27;
  }

  printf("Can fit upto %d cells\n", k);

#warning hard-coded numbers here. probably should be a simulation params...
  real L1 = 3.5;
  int sideLength = (int)sqrt(simState.no_of_cells);
  real3 boxCenter = 0.5*sim_params.box_params.box_max; 
  
  for (long int cellInd = 0; cellInd < simState.no_of_cells; ++cellInd){
      int yCoord = cellInd%sideLength;
      int xCoord = cellInd/sideLength;
      
      for (int nodeInd = 0 ; nodeInd < 180 ; ++nodeInd){
          simState.pos.x.h[cellInd*192+nodeInd] = X[nodeInd] + xCoord*L1 + boxCenter.x;
          simState.pos.y.h[cellInd*192+nodeInd] = Y[nodeInd] + yCoord*L1 + boxCenter.y;
          simState.pos.z.h[cellInd*192+nodeInd] = Z[nodeInd] + boxCenter.z;
      }
  }

  simState.posP.x = simState.pos.x;
  simState.posP.y = simState.pos.y;
  simState.posP.z = simState.pos.z;

  simState.posM.x = simState.pos.x;
  simState.posM.y = simState.pos.y;
  simState.posM.z = simState.pos.z;

  simState.pos.CopyToDevice();
  simState.posP.CopyToDevice();
  simState.posM.CopyToDevice();
return(0);
}


int InitializeRandom(int rand_seed)
{
  // This function uses marsaglia random number generator
  // Defined in marsaglia.h
  int seed_ij, seed_kl ,ij,kl;
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

  rmarin(ij,kl);

  printf("RNG seeds: %d, %d\n", ij, kl); 
  return(0);
}


int read_fullerene_nn(SimState& simState)
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
      simState.C180_nn.h[0 + i] = N1-1;
      simState.C180_nn.h[192+i] = N2-1;
      simState.C180_nn.h[384+i] = N3-1;
      simState.C180_sign.h[i] = Sign;
  }
  fclose(infil);

  if ( end < 180 ) {printf("Error: Unable to read line %d in file C180NN\n",end);return(-1);}

  // printf("   Reading C180C ..\n");

  // infil = fopen("C180C","r");
  // if ( infil == NULL ) {printf("Unable to open file C180C\n");return(-1);}

  // end = 270;
  // for ( i = 0; i < 270 ; ++i )
  // {
  //     if ( fscanf(infil,"%d,%d", &N1, &N2) != 2 ) {end = i; break;}
  //     CCI[0][i] = N1-1;
  //     CCI[1][i] = N2-1;
  // }
  // fclose(infil);

  // if ( end < 270 ) {printf("Error: Unable to read line %d in file C180C\n",end);return(-1);}

  printf("      read nearest neighbour ids for atoms in C180\n");

  printf("   Reading C180 pentagons, hexagons ..\n");

  infil = fopen("C180_pentahexa","r");
  if ( infil == NULL ) {printf("Unable to open file C180_pentahexa\n");return(-1);}

  end = 12;
  for ( i = 0; i < 12 ; ++i )
  {
      if ( fscanf(infil,"%d %d %d %d %d", &N1, &N2, &N3, &N4, &N5) != 5 ) {end = i; break;}
      simState.C180_56.h[i*7+0] = N1;
      simState.C180_56.h[i*7+1] = N2;
      simState.C180_56.h[i*7+2] = N3;
      simState.C180_56.h[i*7+3] = N4;
      simState.C180_56.h[i*7+4] = N5;
      simState.C180_56.h[i*7+5] = N1;
      simState.C180_56.h[i*7+6] = N1;
  }
  if ( end != 12 ) {printf("Error: Unable to read line %d in file C180_pentahexa\n",end);return(-1);}
  end = 80;
  for ( i = 0; i < 80 ; ++i )
  {
      if ( fscanf(infil,"%d %d %d %d %d %d", &N1, &N2, &N3, &N4, &N5, &N6) != 6 ) {end = i; break;}
      simState.C180_56.h[84+i*7+0] = N1;
      simState.C180_56.h[84+i*7+1] = N2;
      simState.C180_56.h[84+i*7+2] = N3;
      simState.C180_56.h[84+i*7+3] = N4;
      simState.C180_56.h[84+i*7+4] = N5;
      simState.C180_56.h[84+i*7+5] = N6;
      simState.C180_56.h[84+i*7+6] = N1;
  }
  if ( end != 80 ) {printf("Error: Unable to read line %d in file C180_pentahexa\n",end);return(-1);}

  fclose(infil);

  printf("Calculating equilibrium bond lengths\n");

  auto X = simState.pos.x.h; 
  auto Y = simState.pos.y.h; 
  auto Z = simState.pos.z.h; 

  for (int i = 0; i < 180; ++i){
      int N1 = simState.C180_nn.h[0 + i];
      int N2 = simState.C180_nn.h[192 + i];
      int N3 = simState.C180_nn.h[384 + i];

      real3 a, b; 
      a = make_real3(X[i], Y[i], Z[i]);

      b = make_real3(X[N1], Y[N1], Z[N1]);
      simState.R0.h[0 + i] = mag(a-b);

      b = make_real3(X[N2], Y[N2], Z[N2]);
      simState.R0.h[192 + i] = mag(a-b);

      b = make_real3(X[N3], Y[N3], Z[N3]);
      simState.R0.h[384 + i] = mag(a-b);
  }

  simState.C180_nn.CopyToDevice();
  simState.C180_sign.CopyToDevice();
  simState.C180_56.CopyToDevice();
  simState.R0.CopyToDevice();

  return(0);
}

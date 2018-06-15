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
#include "SimParams.cuh"
#include "TrajWriter.cuh"
#include "State.cuh"
#include "globals.cuh"
#include "json/json.h"


void CudaFailure();

#define CudaErrorCheck() { \
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

  int* dividingCells; //Cells that are about to divide
  int* totalCells; // No. of cells at every Dividing_steps

  printf("CellDiv version 0.9\n");

  if ( argc !=4 )
  {
      printf("Usage: CellDiv no_of_threads inpFile.json gpuid\n");
      return(0);
  }

  SimState simState(atoi(argv[1]), sim_params);
  char inpFile[256];
  strcpy(inpFile, argv[2]);

  if ( read_json_params(inpFile)          != 0 ) return(-1);

  sim_params_struct sim_params;

  ReadSimParams(sim_params, "inp.json");

  TrajWriter TW(sim_params);

  if ( generate_random(Orig_No_of_C180s)  != 0 ) return(-1);
  if ( initialize_C180s(Orig_No_of_C180s) != 0 ) return(-1);
  if ( read_fullerene_nn()                != 0 ) return(-1);
  NDIV = (int *)calloc(MaxNoofC180s,sizeof(int));
  CPUMemory += MaxNoofC180s*sizeof(int);

  /* PM
     Allocate memory for the dividingCells array that will be used to
     calculate the mitotic index.
  */

  dividingCells = (int *)calloc((Time_steps/newCellCountInt), sizeof(int));
  totalCells = (int *)calloc((Time_steps/newCellCountInt), sizeof(int));
  num_new_cells_per_step = (int *)calloc(Time_steps, sizeof(int));

  cudaDeviceProp deviceProp = getDevice();
  if (cudaSuccess != cudaSetDevice(atoi(argv[3]))){
      printf("Could not set to divice %d\n", 2);
      return -1;
  }
  
    // Better way to see how much GPU memory is being used.
  size_t totalGPUMem;
  size_t freeGPUMem;

  if ( cudaSuccess != cudaMemGetInfo ( &freeGPUMem, &totalGPUMem ) ) {
      printf("Couldn't read GPU Memory status\nExiting...\n");
      CudaErrorCheck();
      exit(1);
  }

  GPUMemory = totalGPUMem - freeGPUMem;

  // Set the Youngs_mod for the cells
  thrust::fill(simState.youngsMod.h.begin(), simState.youngsMod.h.end(),
               sim_params.youngs_mod);

  if (sim_params.use_stiff){
      std::cout << "This hasn't been coded yet :(" << std::endl;
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

  simState.youngsMod.CopyToDevice();

  // initialize device rng
    if (add_rands){
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


  int prevnoofblocks  = simState.no_of_cells;
  int noofblocks      = simState.no_of_cells;
  int threadsperblock = 192;
  printf("   no of blocks = %d, threadsperblock = %d, no of threads = %ld\n",
         noofblocks, threadsperblock, ((long) noofblocks)*((long) threadsperblock));

  CenterOfMass<<<simstate.No_of_C180s,256>>>(simState.devPtrs);
                                             
  
  bounding_boxes<<<No_of_C180s,32>>>(simState.devPtrs);
                                     
  CudaErrorCheck(); 

  size_t reductionblocks = (No_of_C180s-1)/1024+1;
  minmaxpre<<<reductionblocks,1024>>>(simState.devPtrs);
  CudaErrorCheck(); 
  minmaxpost<<<1,1024>>>(simState.devPtrs);
  CudaErrorCheck(); 
  simState.mins.CopyToHost();
  simState.maxs.CopyToHost();
  //  DL = 3.8f;
  DL = 2.9f;
  //DL = divVol; 

  Xdiv = (int)((simState.maxs.x[0]-simState.mins.x[0])/DL + 1);
  Ydiv = (int)((simState.maxs.y[0]-simState.mins.y[0])/DL + 1);
  Zdiv = (int)((simState.maxs.z[0]-simState.mins.z[0])/DL + 1);

  
  makeNNlist<<<No_of_C180s/512+1,512>>>(simState.devPtrs, Xdiv, Ydiv, Zdiv);
  CudaErrorCheck(); 
  globalrank = 0;

  if (correct_com == true){
      CenterOfMass<<<simState.no_of_cells,256>>>(simState.devPtrs);
      simState.cellCOMs.x.CopyToHost();
      simState.cellCOMs.y.CopyToHost();
      simState.cellCOMs.z.CopyToHost();

      real3 sysCM = make_real3(0, 0, 0);

      for (size_t i = 0; i < simState.cellCOMs.x.h.size(), ++i){
          sysCM = sysCM + make_real3(simstate.cellCOMs.x.h[i],
                                     simstate.cellCOMs.y.h[i],
                                     simstate.cellCOMs.z.h[i]);
      }

      sysCM = sysCM/(simState.no_of_cells);
      CorrectCoMMotion<<<(simState.no_of_cells*192)/1024 + 1, 1024>>>(simState.devPtrs, sysCM);
      CudaErrorCheck(); 
  }
  
  

  // Initialize pressures

  thrust::fill(simState.pressures.d.begin(), simState.pressures.d.end(),
               sim_params.min_pressure);
  
  float rGrowth = 0;
  bool growthDone = false;
  
  if (constrainAngles){
      // Code to initialize equillibrium angles
      float3 p, ni, nj, nk;
      R3Nptrs pos = simState.pos.hostPtrs;
      
      for (int n = 0; n<180; n++){
          p = make_float3(pos.x[n], pos.y[n], pos.z[n]); 

          ni = make_float3(pos.x[C180_nn[0*192 + n]], pos.y[C180_nn[0*192 + n]], 
                           pos.z[C180_nn[0*192 + n]]); 
          
          nj = make_float3(pos.x[C180_nn[1*192 + n]], pos.y[C180_nn[1*192 + n]], 
                           pos.z[C180_nn[1*192 + n]]);
          
          nk = make_float3(pos.x[C180_nn[2*192 + n]], pos.y[C180_nn[2*192 + n]],
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

  float r_CM_o = pow((3.0/4.0) * (1/3.14159) * divVol*2.0, 1.0/3);



  printf("   Total amount of GPU memory used =    %8.2lf MB\n",GPUMemory/(1024*1024.0));
  printf("   Total amount of CPU memory used =    %8.2lf MB\n",bas_n::used_mem/(1024*1024.0));

  bool phase = false;

  CalculateConForce<<<No_of_C180s,threadsperblock>>>(simState.devPtrs, sim_params); 
  CudaErrorCheck();

  CalculateDisForce<<<No_of_C180s, threadsperblock>>>(simState.devPtrs, sim_params);
  CudaErrorCheck();


  if (binaryOutput){
      int t = MaxNoofC180s;
      fwrite(&t, sizeof(int), 1, trajfile);
      
      t = (int)useDifferentStiffnesses;
      fwrite(&t, sizeof(int), 1, trajfile);
      
      t = (Time_steps+equiStepCount+1) / trajWriteInt;
      fwrite(&t, sizeof(int), 1, trajfile);
      
      state.no_of_cells = No_of_C180s; 
      WriteBinaryTraj(0, trajfile, 1);
      TW.WriteState(state);
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


  int numNodes = No_of_C180s*192;
  // Simulation loop
  for ( step = 1; step < Time_steps+1 + equiStepCount; step++)
  {
      //printf("step %d\n", step);
      numNodes = No_of_C180s*192;
      Integrate<<<No_of_C180s, threadsperblock>>>(simState.devPtrs, sim_params);
      CudaErrorCheck();

      ForwardTime<<<No_of_C180s, threadsperblock>>>(simState.devPtrs);
      CudaErrorCheck();


      // save previous step forces in g
      
      // if (doPopModel == 1){
      //       rGrowth = rMax * (1 - (No_of_C180s*1.0/maxPop));
      //       // dr = -rGrowth(a + b*rGrowth)
      //       // rGrowth += dr * delta_t ;
      //       // dN/dT = N*R
      //       // dR/dT = -R(a+bR)
      //       // 
      //       if (rGrowth < 0) rGrowth =0; 
      // }
      // else {
      // rGrowth = rMax;
      // }
      PressureUpdate <<<No_of_C180s/1024 + 1, 1024>>> (simState, sim_params, step);
      CudaErrorCheck(); 
      
      if ( (step)%1000 == 0)
      {
          printf("   time %-8d %d cells, rGrowth %f, maxPop %f\n",step,No_of_C180s, rGrowth, maxPop);
      }


#ifdef FORCE_DEBUG
      printf("time %d  pressure = %f\n", step, Pressure);
#endif

      CalculateConForce<<<No_of_C180s,threadsperblock>>>(simState.devPtrs, sim_params, Xdiv, Ydiv, Zdiv);
      CudaErrorCheck();


      CalculateDisForce<<<No_of_C180s, threadsperblock>>>(simSate.devPtrs, sim_params, Xdiv, Ydiv, Zdiv);
      CudaErrorCheck();

      // Calculate random Force here...
      // Placeholder
      
      VelocityUpdateA<<<No_of_C180s, threadsperblock>>>(simState.devPtrs, sim_params);
      CudaErrorCheck();


      // Dissipative velocity update part...
      for (int s = 0; s < 1; ++s){ // may be looped over more later.
          VelocityUpdateB<<<No_of_C180s, threadsperblock>>>(simState.devPtrs, sim_params);
          CudaErrorCheck();

          CalculateDisForce<<<No_of_C180s, threadsperblock>>>(simSate.devPtrs, sim_params, Xdiv, Ydiv, Zdiv);
      CudaErrorCheck();

      // this loop can be looped until convergence, but it shouldn't
      // be necessary most of the time...
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

            PressureReset <<<(2*num_cell_div)/512 + 1, 512>>> (simState.devPtrs, sim_params); 
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

          cudaMemcpy(h_contactForces.x, d_contactForces.x, 192*No_of_C180s*sizeof(float), cudaMemcpyDeviceToHost);
          cudaMemcpy(h_contactForces.y, d_contactForces.y, 192*No_of_C180s*sizeof(float), cudaMemcpyDeviceToHost);
          cudaMemcpy(h_contactForces.z, d_contactForces.z, 192*No_of_C180s*sizeof(float), cudaMemcpyDeviceToHost);
              
          cudaMemcpy(velListX, d_velListX, 192*No_of_C180s*sizeof(float), cudaMemcpyDeviceToHost);
          cudaMemcpy(velListY, d_velListY, 192*No_of_C180s*sizeof(float), cudaMemcpyDeviceToHost);
          cudaMemcpy(velListZ, d_velListZ, 192*No_of_C180s*sizeof(float), cudaMemcpyDeviceToHost);

          cudaMemcpy(state.vol, d_volume, No_of_C180s*sizeof(float), cudaMemcpyDeviceToHost);
          CudaErrorCheck();
          
          state.no_of_cells = No_of_C180s; 
          if (binaryOutput){
              WriteBinaryTraj(step, trajfile, frameCount);
              TW.WriteState(state);
          }
          else
              write_traj(step, trajfile);
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
          printf("Simulation in a box must be with random positions\n");
          exit(12);

          rCheck *= 1.2;
          float3 center = 0.5*boxMax;
          
          for ( rank = 0; rank < Orig_No_of_C180s; ++rank )
          {

              for ( atom = 0 ; atom < 180 ; ++atom)
              {
                  X[rank*192+atom] = initx[atom] + center.x;
                  Y[rank*192+atom] = inity[atom] + center.y;
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

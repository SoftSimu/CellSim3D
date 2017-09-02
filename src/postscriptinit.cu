#include <stdio.h>

#include "Types.cuh"
#include "VectorFunctions.hpp"
#include "State.cuh"

#warning "This source file uses the magic 192 number. Remember to rewrite if that number ever changes!!"

#define CELLDIV_EPS 1e-3

cudaDeviceProp getDevice(void)
{

  int deviceCount;
  cudaGetDeviceCount(&deviceCount);
  int device;
  cudaDeviceProp deviceProp;
  for (device = 0; device < deviceCount; ++device) {
       cudaGetDeviceProperties(&deviceProp, device);
       printf("   Device %s\n", deviceProp.name);
       printf("      compute capability           =         %d.%d\n", deviceProp.major, deviceProp.minor);
       printf("      totalGlobalMemory            =        %.2lf GB\n", deviceProp.totalGlobalMem/1000000000.0);
       printf("      l2CacheSize                  =    %8d\n", deviceProp.l2CacheSize);
       printf("      regsPerBlock                 =    %8d\n", deviceProp.regsPerBlock);
       printf("      multiProcessorCount          =    %8d\n", deviceProp.multiProcessorCount);
       printf("      maxThreadsPerMultiprocessor  =    %8d\n", deviceProp.maxThreadsPerMultiProcessor);
       printf("      sharedMemPerBlock            =    %8d B\n", (int)deviceProp.sharedMemPerBlock);
       printf("      warpSize                     =    %8d\n", deviceProp.warpSize);
       printf("      clockRate                    =    %8.2lf MHz\n", deviceProp.clockRate/1000.0);
       printf("      maxThreadsPerBlock           =    %8d\n", deviceProp.maxThreadsPerBlock);
       printf("      asyncEngineCount             =    %8d\n", deviceProp.asyncEngineCount);
       printf("      concurrentKernels            =    ");
       if(deviceProp.concurrentKernels==1) printf("   yes\n"); else printf("   no\n");

       printf("      ComputeMode                  =    %8d\n", deviceProp.computeMode);
       }

  return deviceProp; 
}



__device__ void CalcAndUpdateDaughtPos(const real3& parNodePos,
                                       const real3& norm,
                                       const real3& parCoM,
                                       const real& halfgap,
                                       SimStatePtrs& sim_state,
                                       const size_t& globalDaughtNodeInd){

    real3 daughtNodePos = parNodePos - parCoM - halfgap*norm;

    real mu = dot(daughtNodePos, norm);

    // if node is below division plane, project on the th division plane
    if (mu <= 0){
        daughtNodePos = daughtNodePos - mu*norm;
    }

    // restore node position back to its original position wrt to the system origin
    daughtNodePos = daughtNodePos + parCoM + halfgap*norm;

    write_to_R3N_state(sim_state.posP, daughtNodePos, globalDaughtNodeInd);
    write_to_R3N_state(sim_state.pos, daughtNodePos, globalDaughtNodeInd);
    write_to_R3N_state(sim_state.posM, daughtNodePos, globalDaughtNodeInd);
}

__global__ void  cell_division(size_t parCellInd, size_t daughtCellInd,
                               SimStatePtrs sim_state, sim_params_struct sim_params, real3 norm){
    int localNodeInd = threadIdx.x;

    if ( localNodeInd < 180 ) 
    {

#ifdef CELLDIV_EPS
#warning "This source file has hard coded constants!!!"
        if (abs(mag(norm)) > CELLDIV_EPS){
            asm("trap;");
        }
#endif

        real halfgap = 0.5*sim_params.core_params.rep_range;

        size_t globalNodeInd = parCellInd*192+localNodeInd;
        size_t globalDaughtNodeInd = daughtCellInd*192+localNodeInd; 
                
        real3 parNodePos = make_real3(sim_state.pos.x[globalNodeInd],
                                      sim_state.pos.y[globalNodeInd],
                                      sim_state.pos.z[globalNodeInd]);

        real3 parCoM = make_real3(sim_state.cellCOMs.x[parCellInd],
                                  sim_state.cellCOMs.y[parCellInd],
                                  sim_state.cellCOMs.z[parCellInd]);

        CalcAndUpdateDaughtPos(parNodePos, norm, parCoM, halfgap, sim_state,
                               globalNodeInd);

        CalcAndUpdateDaughtPos(parNodePos, -1*norm, parCoM, halfgap, sim_state,
                               globalDaughtNodeInd);

        copy_R3N_state(sim_state.vel, globalDaughtNodeInd, globalNodeInd);
    }
}



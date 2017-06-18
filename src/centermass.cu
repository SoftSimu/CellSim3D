#include <cuda.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "postscript.h"
#include "State.cuh"

__global__ void CenterOfMass(SimStatePtrs sim_state)
{
__shared__ float  sumx[256];
__shared__ float  sumy[256];
__shared__ float  sumz[256];

int rank = blockIdx.x;
int tid  = threadIdx.x;

sumx[tid] = 0.0;
sumy[tid] = 0.0;
sumz[tid] = 0.0;

if ( tid < 180 )
    {
    sumx[tid] = sim_state.posP.x[rank*192+tid];
    sumy[tid] = sim_state.posP.y[rank*192+tid];
    sumz[tid] = sim_state.posP.z[rank*192+tid];
    }

__syncthreads();

for ( int s = blockDim.x/2; s > 0; s>>=1)
   {
   if ( tid < s )
      {
      sumx[tid] += sumx[tid+s];
      sumy[tid] += sumy[tid+s];
      sumz[tid] += sumz[tid+s];
      }
   __syncthreads();
   }

if ( tid == 0 ) 
   {
   sim_state.cellCOMs.x[rank] = sumx[0]/180.0f;
   sim_state.cellCOMs.y[rank] = sumy[0]/180.0f;
   sim_state.cellCOMs.z[rank] = sumz[0]/180.0f;
   }

}

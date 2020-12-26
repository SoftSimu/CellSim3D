#include <cuda.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "postscript.h"


__global__ void CenterOfMass( int No_of_C180s, 
               float *d_XP, float *d_YP, float *d_ZP, 
               float *CMx, float *CMy,float *CMz)
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
    sumx[tid] = d_XP[rank*192+tid];
    sumy[tid] = d_YP[rank*192+tid];
    sumz[tid] = d_ZP[rank*192+tid];
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
   CMx[rank] = sumx[0]/180.0f;
   CMy[rank] = sumy[0]/180.0f;
   CMz[rank] = sumz[0]/180.0f;
   }

}




__global__ void VelocityCenterOfMass( int No_of_C180s, 
               float *d_velListX, float *d_velListY, float *d_velListZ, 
               float *d_VCMx, float *d_VCMy,float *d_VCMz)
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
    sumx[tid] = d_velListX[rank*192+tid];
    sumy[tid] = d_velListY[rank*192+tid];
    sumz[tid] = d_velListZ[rank*192+tid];
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
   d_VCMx[rank] = sumx[0]/180.0f;
   d_VCMy[rank] = sumy[0]/180.0f;
   d_VCMz[rank] = sumz[0]/180.0f;
   }

}






#include <cuda.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "postscript.h"


__global__ void CenterOfMass( int No_of_C180s, 
               float *d_X, float *d_Y, float *d_Z, 
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
    sumx[tid] = d_X[rank*192+tid];
    sumy[tid] = d_Y[rank*192+tid];
    sumz[tid] = d_Z[rank*192+tid];
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


__global__ void SysCMpost( int No_of_C180s, float *d_Cx, float *d_Cy,float *d_Cz, 
			   float* SysCx, float* SysCy, float* SysCz)
{

	__shared__ float  sumx[1024];
	__shared__ float  sumy[1024];
	__shared__ float  sumz[1024];


  	int fullerene = blockIdx.x*blockDim.x+threadIdx.x;
	int tid = threadIdx.x;
	int rank = blockIdx.x;
	  
	sumx[tid] = 0.0;
	sumy[tid] = 0.0;
	sumz[tid] = 0.0;

	if ( fullerene < No_of_C180s )
    	{
    		sumx[tid] = d_Cx[fullerene];
    		sumy[tid] = d_Cy[fullerene];
    		sumz[tid] = d_Cz[fullerene];
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
	      
	      SysCx[rank]  = sumx[0];
	      SysCy[rank]  = sumy[0];
	      SysCz[rank]  = sumz[0];

	}

}


__global__ void SysCM( int No_of_C180s, int reductionblocks,
			float* SysCx, float* SysCy, float* SysCz,
			R3Nptrs d_sysCM)
{

	__shared__ float  sysCmx[1024];
  	__shared__ float  sysCmy[1024];
  	__shared__ float  sysCmz[1024];


  	int tid = threadIdx.x;

	sysCmx[tid] = 0.0;
	sysCmy[tid] = 0.0;
	sysCmz[tid] = 0.0;


	if ( tid < reductionblocks )
    	{
    		sysCmx[tid] = SysCx[tid];
    		sysCmy[tid] = SysCy[tid];
    		sysCmz[tid] = SysCz[tid];
    	}

        __syncthreads();


        for ( int s = blockDim.x/2; s > 0; s>>=1)
   	{
   		if ( tid < s )
      		{
      			sysCmx[tid] += sysCmx[tid+s];
      			sysCmy[tid] += sysCmy[tid+s];
      			sysCmz[tid] += sysCmz[tid+s];
      		}
   		__syncthreads();
   	}


  	if ( tid == 0 )
	{
		*d_sysCM.x = sysCmx[0]/No_of_C180s;
		*d_sysCM.y = sysCmy[0]/No_of_C180s;
		*d_sysCM.z = sysCmz[0]/No_of_C180s;
	}

}




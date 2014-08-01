#include <cuda.h>
#include <stdio.h>
#include <math.h>

__global__ void volumes( int No_of_C180s, int *C180_56, 
                         float *X,    float *Y,   float *Z, 
                         float *CMx , float *CMy, float *CMz, float *vol,
                         char* cell_div)
{
__shared__ float locX[192];
__shared__ float locY[192];
__shared__ float locZ[192];
__shared__ float volume;

int fullerene = blockIdx.x;
int tid       = threadIdx.x;

if ( tid < 180 ) 
   {
   locX[tid] = X[192*fullerene+tid] -CMx[fullerene];
   locY[tid] = Y[192*fullerene+tid] -CMy[fullerene];
   locZ[tid] = Z[192*fullerene+tid] -CMz[fullerene];
   }

if ( tid == 0) volume = 0.0f;

__syncthreads();

if ( tid < 92 )
   {

   float avX = 0.0f;
   float avY = 0.0f;
   float avZ = 0.0f;

   for ( int i = 0; i < 5; ++i )
       {
       avX += locX[C180_56[7*tid+i]];
       avY += locY[C180_56[7*tid+i]];
       avZ += locZ[C180_56[7*tid+i]];
       }

  float avefactor = 0.166666667f;
  if ( tid < 12 ) 
       {
       avefactor = 0.2f;
       }
  else
       {
       avX += locX[C180_56[7*tid+5]];
       avY += locY[C180_56[7*tid+5]];
       avZ += locZ[C180_56[7*tid+5]];
       }

   avX *= avefactor;
   avY *= avefactor;
   avZ *= avefactor;

   float totvol = 0.0f;
   float n1 = 0.0f;
   float n2 = 0.0f;
   float n3 = 0.0f;

   for ( int i = 0; i < 6; ++i )
       {
       n1 = (locY[C180_56[7*tid+i+1]]*avZ-avY*locZ[C180_56[7*tid+i+1]])*locX[C180_56[7*tid+i]];
       n2 = (locZ[C180_56[7*tid+i+1]]*avX-avZ*locX[C180_56[7*tid+i+1]])*locY[C180_56[7*tid+i]];
       n3 = (locX[C180_56[7*tid+i+1]]*avY-avX*locY[C180_56[7*tid+i+1]])*locZ[C180_56[7*tid+i]];
       totvol += fabsf(n1+n2+n3);
       }

    atomicAdd(&volume, totvol);
    }

__syncthreads();

 if ( tid == 0){
   vol[fullerene] = volume/6.0;
   if ((volume/6.0) > 2.9f){
     cell_div[fullerene] = 1;
   }
 }

}


#include <cuda.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "VectorFunctions.hpp"
#include "State.cuh"

//#define PRINT_TOO_SHORT_ERROR

__global__ void bounding_boxes(SimStatePtrs sim_state)
{
  __shared__ float  minx[32];
  __shared__ float  maxx[32];
  __shared__ float  miny[32];
  __shared__ float  maxy[32];
  __shared__ float  minz[32];
  __shared__ float  maxz[32];

  __shared__ real* d_XP; 
  __shared__ real* d_YP; 
  __shared__ real* d_ZP;

  int rank = blockIdx.x;
  int tid  = threadIdx.x;
  int atom = tid;

  if (tid == 0){
      d_XP = sim_state.posP.x;
      d_YP = sim_state.posP.y;
      d_ZP = sim_state.posP.z;
  }
  __syncthreads();

  if ( rank < sim_state.no_of_cells )
    {
	  minx[tid] = d_XP[rank*192+atom];
	  maxx[tid] = d_XP[rank*192+atom];
	  miny[tid] = d_YP[rank*192+atom];
	  maxy[tid] = d_YP[rank*192+atom];
	  minz[tid] = d_ZP[rank*192+atom];
	  maxz[tid] = d_ZP[rank*192+atom];

	  while ( atom + 32 < 180 )
        {
		  atom += 32;
		  if ( minx[tid] > d_XP[rank*192+atom] )
		       minx[tid] = d_XP[rank*192+atom];
		  if ( maxx[tid] < d_XP[rank*192+atom] )
		       maxx[tid] = d_XP[rank*192+atom];
		  if ( miny[tid] > d_YP[rank*192+atom] )
		       miny[tid] = d_YP[rank*192+atom];
		  if ( maxy[tid] < d_YP[rank*192+atom] )
		       maxy[tid] = d_YP[rank*192+atom];
		  if ( minz[tid] > d_ZP[rank*192+atom] )
		       minz[tid] = d_ZP[rank*192+atom];
		  if ( maxz[tid] < d_ZP[rank*192+atom] )
		       maxz[tid] = d_ZP[rank*192+atom];
        }

	  if ( tid < 16 )
        {
		  if ( minx[tid] > minx[tid+16] ) minx[tid] = minx[tid+16];
		  if ( maxx[tid] < maxx[tid+16] ) maxx[tid] = maxx[tid+16];

		  if ( miny[tid] > miny[tid+16] ) miny[tid] = miny[tid+16];
		  if ( maxy[tid] < maxy[tid+16] ) maxy[tid] = maxy[tid+16];

		  if ( minz[tid] > minz[tid+16] ) minz[tid] = minz[tid+16];
		  if ( maxz[tid] < maxz[tid+16] ) maxz[tid] = maxz[tid+16];
        }

	  if ( tid < 8 )
        {
		  if ( minx[tid] > minx[tid+8] ) minx[tid] = minx[tid+8];
		  if ( maxx[tid] < maxx[tid+8] ) maxx[tid] = maxx[tid+8];
		  if ( miny[tid] > miny[tid+8] ) miny[tid] = miny[tid+8];
		  if ( maxy[tid] < maxy[tid+8] ) maxy[tid] = maxy[tid+8];
		  if ( minz[tid] > minz[tid+8] ) minz[tid] = minz[tid+8];
		  if ( maxz[tid] < maxz[tid+8] ) maxz[tid] = maxz[tid+8];
        }

	  if ( tid < 4 )
        {
		  if ( minx[tid] > minx[tid+4] ) minx[tid] = minx[tid+4];
		  if ( maxx[tid] < maxx[tid+4] ) maxx[tid] = maxx[tid+4];
		  if ( miny[tid] > miny[tid+4] ) miny[tid] = miny[tid+4];
		  if ( maxy[tid] < maxy[tid+4] ) maxy[tid] = maxy[tid+4];
		  if ( minz[tid] > minz[tid+4] ) minz[tid] = minz[tid+4];
		  if ( maxz[tid] < maxz[tid+4] ) maxz[tid] = maxz[tid+4];
        }

	  if ( tid < 2 )
        {
		  if ( minx[tid] > minx[tid+2] ) minx[tid] = minx[tid+2];
		  if ( maxx[tid] < maxx[tid+2] ) maxx[tid] = maxx[tid+2];
		  if ( miny[tid] > miny[tid+2] ) miny[tid] = miny[tid+2];
		  if ( maxy[tid] < maxy[tid+2] ) maxy[tid] = maxy[tid+2];
		  if ( minz[tid] > minz[tid+2] ) minz[tid] = minz[tid+2];
		  if ( maxz[tid] < maxz[tid+2] ) maxz[tid] = maxz[tid+2];
        }

	  if ( tid == 0  )
        {
            real* d_bounding_xyz = sim_state.boundingBoxes;
		  if ( minx[0] > minx[1] ) minx[0] = minx[1];
		  d_bounding_xyz[rank*6+0] = minx[0];

		  if ( maxx[0] < maxx[1] ) maxx[0] = maxx[1];
		  d_bounding_xyz[rank*6+1] = maxx[0];

		  if ( miny[0] > miny[1] ) miny[0] = miny[1];
		  d_bounding_xyz[rank*6+2] = miny[0];

		  if ( maxy[0] < maxy[1] ) maxy[0] = maxy[1];
		  d_bounding_xyz[rank*6+3] = maxy[0];

		  if ( minz[0] > minz[1] ) minz[0] = minz[1];
		  d_bounding_xyz[rank*6+4] = minz[0];

		  if ( maxz[0] < maxz[1] ) maxz[0] = maxz[1];
		  d_bounding_xyz[rank*6+5] = maxz[0];
        }
    }
}



__global__ void minmaxpre( SimStatePtrs sim_state)
{

  __shared__ float  minx[1024];
  __shared__ float  maxx[1024];
  __shared__ float  miny[1024];
  __shared__ float  maxy[1024];
  __shared__ float  minz[1024];
  __shared__ float  maxz[1024];
  __shared__ real* d_bounding_xyz;
  
  if (threadIdx.x == 0){
      d_bounding_xyz = sim_state.boundingBoxes;
  }

  int fullerene = blockIdx.x*blockDim.x+threadIdx.x;
  int tid       = threadIdx.x;

  minx[tid] = +1.0E8f;
  maxx[tid] = -1.0E8f;
  miny[tid] = +1.0E8f;
  maxy[tid] = -1.0E8f;
  minz[tid] = +1.0E8f;
  maxz[tid] = -1.0E8f;

  if ( fullerene < sim_state.no_of_cells)
    {
	  minx[tid] = d_bounding_xyz[6*fullerene+0];
	  maxx[tid] = d_bounding_xyz[6*fullerene+1];
	  miny[tid] = d_bounding_xyz[6*fullerene+2];
	  maxy[tid] = d_bounding_xyz[6*fullerene+3];
	  minz[tid] = d_bounding_xyz[6*fullerene+4];
	  maxz[tid] = d_bounding_xyz[6*fullerene+5];
    }

  __syncthreads();

  for ( int s = blockDim.x/2; s > 0; s>>=1)
	{
	  if ( tid < s )
		{
		  minx[tid] = fminf(minx[tid],minx[tid+s]);
		  maxx[tid] = fmaxf(maxx[tid],maxx[tid+s]);
		  miny[tid] = fminf(miny[tid],miny[tid+s]);
		  maxy[tid] = fmaxf(maxy[tid],maxy[tid+s]);
		  minz[tid] = fminf(minz[tid],minz[tid+s]);
		  maxz[tid] = fmaxf(maxz[tid],maxz[tid+s]);
		}
	  __syncthreads();
	}

  if ( tid == 0 )
	{
	  // Minx[blockIdx.x]  = minx[0];
	  // Maxx[blockIdx.x]  = maxx[0];
	  // Miny[blockIdx.x]  = miny[0];
	  // Maxy[blockIdx.x]  = maxy[0];
	  // Minz[blockIdx.x]  = minz[0];
	  // Maxz[blockIdx.x]  = maxz[0];

      sim_state.mins.x[blockIdx.x] = minx[0];
      sim_state.mins.y[blockIdx.x] = miny[0];
      sim_state.mins.z[blockIdx.x] = minz[0];

      sim_state.maxs.x[blockIdx.x] = maxx[0];
      sim_state.maxs.y[blockIdx.x] = maxy[0];
      sim_state.maxs.z[blockIdx.x] = maxz[0];
	}

}




__global__ void makeNNlist(SimStatePtrs sim_state, sim_params_struct sm, int Xdiv, int Ydiv, int Zdiv){


  int fullerene = blockIdx.x*blockDim.x+threadIdx.x;
  
  real* d_bounding_xyz = sim_state.boundingBoxes;
  real attrac = sm.core_params.attr_range;
  real DL = sm.core_params.dom_len;
  real Minx = sim_state.mins.x[0];
  real Miny = sim_state.mins.y[0];
  real Minz = sim_state.mins.z[0];

  if ( fullerene < sim_state.no_of_cells )
	{

	  int startx = (int)((d_bounding_xyz[6*fullerene+0]-attrac
						  - Minx)/DL);
	  if ( startx < 0 ) startx = 0;
	  int endx   = (int)((d_bounding_xyz[6*fullerene+1]+attrac
						  - Minx)/DL);
	  if ( endx >= Xdiv ) endx = Xdiv-1;
	  int starty = (int)((d_bounding_xyz[6*fullerene+2]-attrac
						  - Miny)/DL);
	  if ( starty < 0 ) starty = 0;
	  int  endy  = (int)((d_bounding_xyz[6*fullerene+3]+attrac
						  - Miny)/DL);
	  if ( endy >= Ydiv ) endy = Ydiv-1;
	  int startz = (int)((d_bounding_xyz[6*fullerene+4]-attrac
						  - Minz)/DL);
	  if ( startz < 0 ) startz = 0;
	  int  endz  = (int)((d_bounding_xyz[6*fullerene+5]+attrac
						  - Minz)/DL);
	  if ( endz >= Zdiv ) endz = Zdiv-1;

	  for ( int j1 = startx; j1 <= endx; ++j1 )
              for ( int j2 = starty; j2 <= endy; ++j2 )
		  for ( int j3 = startz; j3 <= endz; ++j3 )
			{
                int index = atomicAdd( &(sim_state.numOfNNList[j3*Xdiv*Ydiv+j2*Xdiv+j1]) , 1); //returns old
#ifdef PRINT_TOO_SHORT_ERROR
			  if ( index > 32 )
				{
                                    printf("Fullerene %d, NN-list too short, atleast %d\n", fullerene, index);
                                  // for ( int k = 0; k < 32; ++k )
                                  //     printf("%d ",d_NNlist[ 32*(j2*Xdiv+j1) + k]);
                                  
                                  // printf("\n");
				  continue;
				}
#endif
			  sim_state.nnList[ 32*(j3*Xdiv*Ydiv+j2*Xdiv+j1)+index] = fullerene;
			}
	}

}


__global__ void minmaxpost(SimStatePtrs sim_state){

    __shared__ float  minx[1024];
    __shared__ float  maxx[1024];
    __shared__ float  miny[1024];
    __shared__ float  maxy[1024];
    __shared__ float  minz[1024];
    __shared__ float  maxz[1024];

    int fullerene = blockIdx.x*blockDim.x+threadIdx.x;
    int tid       = threadIdx.x;

    minx[tid] = +1.0E8f;
    maxx[tid] = -1.0E8f;
    miny[tid] = +1.0E8f;
    maxy[tid] = -1.0E8f;
    minz[tid] = +1.0E8f;
    maxz[tid] = -1.0E8f;

    real* Minx = sim_state.mins.x; 
    real* Miny = sim_state.mins.x; 
    real* Minz = sim_state.mins.x; 

    real* Maxx = sim_state.maxs.x; 
    real* Maxy = sim_state.maxs.x; 
    real* Maxz = sim_state.maxs.x; 

    if ( fullerene < sim_state.no_of_cells ){
        minx[tid] = Minx[fullerene];
        maxx[tid] = Maxx[fullerene];
        miny[tid] = Miny[fullerene];
        maxy[tid] = Maxy[fullerene];
        minz[tid] = Minz[fullerene];
        maxz[tid] = Maxz[fullerene];
    }

    __syncthreads();

    for ( int s = blockDim.x/2; s > 0; s>>=1){
        if ( tid < s ){
            minx[tid] = fminf(minx[tid],minx[tid+s]);
            maxx[tid] = fmaxf(maxx[tid],maxx[tid+s]);
            miny[tid] = fminf(miny[tid],miny[tid+s]);
            maxy[tid] = fmaxf(maxy[tid],maxy[tid+s]);
            minz[tid] = fminf(minz[tid],minz[tid+s]);
            maxz[tid] = fmaxf(maxz[tid],maxz[tid+s]);
		}
        __syncthreads();
	}

    if ( tid == 0 )
	{
        // Minx[blockIdx.x+0]  = minx[0];
        // Minx[blockIdx.x+1]  = maxx[0];
        // Minx[blockIdx.x+2]  = miny[0];
        // Minx[blockIdx.x+3]  = maxy[0];
        // Minx[blockIdx.x+4]  = minz[0];
        // Minx[blockIdx.x+5]  = maxz[0];

        sim_state.mins.x[blockIdx.x] = minx[0];
        sim_state.mins.y[blockIdx.x] = miny[0];
        sim_state.mins.z[blockIdx.x] = minz[0];

        sim_state.maxs.x[blockIdx.x] = minx[0];
        sim_state.maxs.y[blockIdx.x] = miny[0];
        sim_state.maxs.z[blockIdx.x] = minz[0];
	}
}

#include <cuda.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "postscript.h"
#include "VectorFunctions.hpp"

//#define PRINT_TOO_SHORT_ERROR

__global__ void bounding_boxes( int No_of_C180s,
               float *d_XP, float *d_YP, float *d_ZP,
//               float *d_X,  float *d_Y,  float *d_Z,
//               float *d_XM, float *d_YM, float *d_ZM,
               float *d_bounding_xyz, float *CMx,
			   float *CMy, float *CMz)
{
  __shared__ float  minx[32];
  __shared__ float  maxx[32];
  __shared__ float  miny[32];
  __shared__ float  maxy[32];
  __shared__ float  minz[32];
  __shared__ float  maxz[32];

  int rank = blockIdx.x;
  int tid  = threadIdx.x;
  int atom = tid;

  if ( rank < No_of_C180s )
    {
	  minx[tid] = d_XP[rank*192+atom];
	  maxx[tid] = d_XP[rank*192+atom];
	  miny[tid] = d_YP[rank*192+atom];
	  maxy[tid] = d_YP[rank*192+atom];
	  minz[tid] = d_ZP[rank*192+atom];
	  maxz[tid] = d_ZP[rank*192+atom];

	  // // move present value to past value
	  // d_XM[rank*192+atom] =  d_X[rank*192+atom];
	  // d_YM[rank*192+atom] =  d_Y[rank*192+atom];
	  // d_ZM[rank*192+atom] =  d_Z[rank*192+atom];

	  // // move future value to present value
	  // d_X[rank*192+atom] = d_XP[rank*192+atom];
	  // d_Y[rank*192+atom] = d_YP[rank*192+atom];
	  // d_Z[rank*192+atom] = d_ZP[rank*192+atom];


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

		  // // move present value to past value
		  // d_XM[rank*192+atom] =  d_X[rank*192+atom];
		  // d_YM[rank*192+atom] =  d_Y[rank*192+atom];
		  // d_ZM[rank*192+atom] =  d_Z[rank*192+atom];

		  // // move future value to present value
		  // d_X[rank*192+atom]  = d_XP[rank*192+atom];
		  // d_Y[rank*192+atom]  = d_YP[rank*192+atom];
		  // d_Z[rank*192+atom]  = d_ZP[rank*192+atom];

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



__global__ void minmaxpre( int No_of_C180s, float *d_bounding_xyz,
                    float *Minx, float *Maxx,
					float *Miny, float *Maxy,
					float *Minz, float *Maxz)
{

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

  if ( fullerene < No_of_C180s )
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
	  Minx[blockIdx.x]  = minx[0];
	  Maxx[blockIdx.x]  = maxx[0];
	  Miny[blockIdx.x]  = miny[0];
	  Maxy[blockIdx.x]  = maxy[0];
	  Minz[blockIdx.x]  = minz[0];
	  Maxz[blockIdx.x]  = maxz[0];
	}

}



__global__ void minmaxpost( int No_of_C180s,
							float *Minx, float *Maxx, float *Miny, float *Maxy,  float *Minz, float *Maxz)
{

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

  if ( fullerene < No_of_C180s )
    {
	  minx[tid] = Minx[fullerene];
	  maxx[tid] = Maxx[fullerene];
	  miny[tid] = Miny[fullerene];
	  maxy[tid] = Maxy[fullerene];
	  minz[tid] = Minz[fullerene];
	  maxz[tid] = Maxz[fullerene];
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
	  Minx[blockIdx.x+0]  = minx[0];
	  Minx[blockIdx.x+1]  = maxx[0];
	  Minx[blockIdx.x+2]  = miny[0];
	  Minx[blockIdx.x+3]  = maxy[0];
	  Minx[blockIdx.x+4]  = minz[0];
	  Minx[blockIdx.x+5]  = maxz[0];
	}

}



__global__ void makeNNlist(int No_of_C180s, float *CMx, float *CMy,float *CMz, float *CMxNNlist, float *CMyNNlist, float *CMzNNlist,
                           int Xdiv, int Ydiv, int Zdiv, float3 BoxMin,
                           int *d_NoofNNlist, int *d_NNlist, float DL)
{


	int fullerene = blockIdx.x*blockDim.x+threadIdx.x;
//  printf("(%d, %d, %d) %d %d\n", blockIdx.x, blockDim.x, threadIdx.x, fullerene, No_of_C180s);


	if ( fullerene < No_of_C180s )
	{
	  
		int posx = 0;
		int posy = 0;
		int posz = 0;		
		

	 	posx = (int)((CMx[fullerene] - BoxMin.x)/DL);
	  	if ( posx < 0 ) posx = 0;
	  	if ( posx > Xdiv - 1 ) posx = Xdiv - 1;
	  	

	  	posy = (int)((CMy[fullerene]-BoxMin.y)/DL);
	  	if ( posy < 0 ) posy = 0;
	  	if ( posy > Ydiv - 1 ) posy = Ydiv - 1;

	   	posz = (int)((CMz[fullerene]-BoxMin.z)/DL);
	  	if ( posz < 0 ) posz = 0;
	  	if ( posz > Zdiv - 1 ) posz = Zdiv - 1;
	  	
	 
		int j1 = 0;
	  	int j2 = 0;
	  	int j3 = 0;
	 
	  	for (  int i = -1; i < 2 ; ++i ){
				
			j1 = posx + i;
			if(j1 < 0 || j1 > Xdiv-1) continue;
			

			for (  int j = -1; j < 2; ++j ){
					
				j2 = posy + j;
				if(j2 < 0 || j2 > Ydiv-1) continue;
				
	
				for (  int k = -1 ; k < 2; ++k ){
			
					j3 = posz + k;
					if(j3 < 0 || j3 > Zdiv-1) continue;
		

			  		int index = atomicAdd( &d_NoofNNlist[j3*Xdiv*Ydiv+j2*Xdiv+j1] , 1); //returns old
#ifdef PRINT_TOO_SHORT_ERROR
			  		if ( index > 64 )
					{
                         			printf("Fullerene %d, NN-list too short, atleast %d\n", fullerene, index);
                                  			// for ( int k = 0; k < 32; ++k )
                                  			//     printf("%d ",d_NNlist[ 32*(j2*Xdiv+j1) + k]); 
                                 			// printf("\n");
						 continue;
					}
#endif
			  		d_NNlist[ 64*(j3*Xdiv*Ydiv+j2*Xdiv+j1)+index] = fullerene;
					
				}
	
			}
		}	
		
	
		
		CMxNNlist[fullerene] = CMx[fullerene];
		CMyNNlist[fullerene] = CMy[fullerene];
		CMzNNlist[fullerene] = CMz[fullerene];
	
	
	}

}

__global__ void makeNNlistMultiGpu( int No_of_C180s, float *d_CMx, float *d_CMy,float *d_CMz, float *CMxNNlist, float *CMyNNlist, float *CMzNNlist,
                           int Xdiv, int Ydiv, int Zdiv, float3 Subdivision_min, float3 Subdivision_max, float3 BoxMin, float3 boxMax,
                           int *d_NoofNNlist, int *d_NNlist, float DL, int* d_counter_gc_e, int* d_counter_gc_w,
                           int* d_counter_gc_n, int* d_counter_gc_s, int* d_counter_gc_u, int* d_counter_gc_d,
                           int* d_Ghost_Cells_ind_EAST, int* d_Ghost_Cells_ind_WEST, int* d_Ghost_Cells_ind_NORTH, int* d_Ghost_Cells_ind_SOUTH,
                           int* d_Ghost_Cells_ind_UP, int* d_Ghost_Cells_ind_DOWN )
{


	int fullerene = blockIdx.x*blockDim.x+threadIdx.x;
//  printf("(%d, %d, %d) %d %d\n", blockIdx.x, blockDim.x, threadIdx.x, fullerene, No_of_C180s);


	if ( fullerene < No_of_C180s )
	{	
		
		
		float Cx = d_CMx[fullerene];
		float Cy = d_CMy[fullerene];
		float Cz = d_CMz[fullerene];
		
	 	int posx = (int)((Cx - Subdivision_min.x)/DL);
	  	if ( posx < 0 ) posx = 0;
	  	if ( posx > Xdiv - 1 ) posx = Xdiv - 1;
	  	

	  	int posy = (int)((Cy - Subdivision_min.y)/DL);
	  	if ( posy < 0 ) posy = 0;
	  	if ( posy > Ydiv - 1 ) posy = Ydiv - 1;

	   	int posz = (int)((Cz - Subdivision_min.z)/DL);
	  	if ( posz < 0 ) posz = 0;
	  	if ( posz > Zdiv - 1 ) posz = Zdiv - 1;
	  	 
	 
	 
	 		
	 	//if (Subdivision_max.x < boxMax.x) {	
	 		
	 	if( Cx >=  Subdivision_max.x - 2.0){
	 			
	 		int index = atomicAdd(d_counter_gc_e,1);
	 		d_Ghost_Cells_ind_EAST[index] = fullerene;	 			
	 	}
	 	
	 	//}
	 	
	 	//if ( BoxMin.x < Subdivision_min.x){
	 	
	 	if( Cx <=  Subdivision_min.x + 2.0 ){
	 			
	 		int index = atomicAdd(d_counter_gc_w,1);
	 		d_Ghost_Cells_ind_WEST[index] = fullerene;
	 			//if (rank == 1) printf("Xpos is:	%f\n", Cx);
	 	}
	 	
	 	//}
	 	
	 	//if (Subdivision_max.y < boxMax.y) {	
	 		
	 	if( Cy >=  Subdivision_max.y - 2.0 ){
	 			
	 		int index = atomicAdd(d_counter_gc_n,1);
	 		d_Ghost_Cells_ind_NORTH[index] = fullerene;	 			
	 	}
	 	//}
	 	
	 	
	 	//if ( BoxMin.y < Subdivision_min.y){
	 		
	 	if( Cy <=  Subdivision_min.y + 2.0 ){
	 			
	 		int index = atomicAdd(d_counter_gc_s,1);
	 		d_Ghost_Cells_ind_SOUTH[index] = fullerene;
	 	}
	 	//}
	 	
	 	//if (Subdivision_max.z < boxMax.z) {	
	 		
	 	if( Cz >=  Subdivision_max.z - 2.0 ){
	 			
	 		int index = atomicAdd(d_counter_gc_u,1);
	 		d_Ghost_Cells_ind_UP[index] = fullerene;
	 				 			
	 	}
	 	//}
	 	
	 	//if ( BoxMin.z < Subdivision_min.z){
	 	if( Cz <= Subdivision_min.z + 2.0 ){
	 			
	 		int index = atomicAdd(d_counter_gc_d,1);
	 		d_Ghost_Cells_ind_DOWN[index] = fullerene;
	 	}	
	 	//}
	 	
	 	
	 
		int j1 = 0;
	  	int j2 = 0;
	  	int j3 = 0;
	 
	  	for (  int i = -1; i < 2 ; ++i ){
				
			j1 = posx + i;
			if(j1 < 0 || j1 > Xdiv-1) continue;
			

			for (  int j = -1; j < 2; ++j ){
					
				j2 = posy + j;
				if(j2 < 0 || j2 > Ydiv-1) continue;
				
	
				for (  int k = -1 ; k < 2; ++k ){
			
					j3 = posz + k;
					if(j3 < 0 || j3 > Zdiv-1) continue;
		

			  		int index = atomicAdd( &d_NoofNNlist[j3*Xdiv*Ydiv+j2*Xdiv+j1] , 1); //returns old
#ifdef PRINT_TOO_SHORT_ERROR
			  		if ( index > 64 )
					{
                         			printf("Fullerene %d, NN-list too short, atleast %d\n", fullerene, index);
                                  			// for ( int k = 0; k < 32; ++k )
                                  			//     printf("%d ",d_NNlist[ 32*(j2*Xdiv+j1) + k]); 
                                 			// printf("\n");
						 continue;
					}
#endif
			  		d_NNlist[ 64*(j3*Xdiv*Ydiv+j2*Xdiv+j1)+index] = fullerene;
					
				}
	
			}
		}	
			
		
		CMxNNlist[fullerene] = d_CMx[fullerene];
		CMyNNlist[fullerene] = d_CMy[fullerene];
		CMzNNlist[fullerene] = d_CMz[fullerene];
	
	
	}

}




__global__ void makeNNlistPBC(int No_of_C180s, float *CMx, float *CMy,float *CMz, float *CMxNNlist, float *CMyNNlist, float *CMzNNlist,
                           float attrac, int Xdiv, int Ydiv, int Zdiv, float3 boxMax,
                           int *d_NoofNNlist, int *d_NNlist, float3 DLp, bool useRigidBoxZ, bool useRigidBoxY)
{


	int fullerene = blockIdx.x*blockDim.x+threadIdx.x;
//  printf("(%d, %d, %d) %d %d\n", blockIdx.x, blockDim.x, threadIdx.x, fullerene, No_of_C180s);


	if ( fullerene < No_of_C180s )
	{
	  
		int posx = 0;
		int posy = 0;
		int posz = 0;	
		float boxX = boxMax.x;
		float boxY = boxMax.y;
		float boxZ = boxMax.z;	
		float DX = DLp.x;
		float DY = DLp.y;
		float DZ = DLp.z;	

	  	
	  	posx = (int)(( CMx[fullerene] - floor( CMx[fullerene] / boxX) * boxX )/DX); 	
 	  	posy = (int)(( CMy[fullerene] - floor( CMy[fullerene] / boxY) * boxY )/DY); 	
	  	posz = (int)(( CMz[fullerene] - floor( CMz[fullerene] / boxZ) * boxZ )/DZ); 		
	 
		int j1 = 0;
	  	int j2 = 0;
	  	int j3 = 0;
	 
	  	for (  int i = -1; i < 2 ; ++i ){
				
			j1 = posx + i;
			j1 = j1 - floor((float)j1/(float)Xdiv) * Xdiv;	 

			for (  int j = -1; j < 2; ++j ){ 


				j2 = posy + j;
					
				if(useRigidBoxY){
					
					if(j2 < 0 || j2 > Ydiv-1) continue;
					
				}else{	
					
					j2 = j2 - floor((float)j2/(float)Ydiv) * Ydiv;	 
					
				}
	
				for (  int k = -1 ; k < 2; ++k ){
			
			
					j3 = posz + k;
					
					if(useRigidBoxZ){
					
						if(j3 < 0 || j3 > Zdiv-1) continue;
					
					}else{	
					
						j3 = j3 - floor((float)j3/(float)Zdiv) * Zdiv;	 
					
					}

			  		int index = atomicAdd( &d_NoofNNlist[j3*Xdiv*Ydiv+j2*Xdiv+j1] , 1); //returns old
#ifdef PRINT_TOO_SHORT_ERROR
			  		if ( index > 64 )
					{
                         			printf("Fullerene %d, NN-list too short, atleast %d\n", fullerene, index);
                                  		// for ( int k = 0; k < 32; ++k )
                                  		// printf("%d ",d_NNlist[ 32*(j2*Xdiv+j1) + k]);
                                  		// printf("\n");
						continue;
					}
#endif
			  		d_NNlist[ 64*(j3*Xdiv*Ydiv+j2*Xdiv+j1)+index] = fullerene;
					
				}
	
			}
		}	

		
		CMxNNlist[fullerene] = CMx[fullerene];
		CMyNNlist[fullerene] = CMy[fullerene];
		CMzNNlist[fullerene] = CMz[fullerene];
		
		
	}
		
}



__global__ void makeNNlistLEbc(int No_of_C180s, float *CMx, float *CMy,float *CMz, float *CMxNNlist, float *CMyNNlist, float *CMzNNlist,
                           float attrac, int Xdiv, int Ydiv, int Zdiv, float3 boxMax,
                           int *d_NoofNNlist, int *d_NNlist, float3 DLp, float Pshift,bool useRigidBoxZ)
{


	int fullerene = blockIdx.x*blockDim.x+threadIdx.x;
//  printf("(%d, %d, %d) %d %d\n", blockIdx.x, blockDim.x, threadIdx.x, fullerene, No_of_C180s);


	if ( fullerene < No_of_C180s )
	{
	  
		int posx = 0;
		int posy = 0;
		int posz = 0;		

	  	
	  	posx = (int)(( CMx[fullerene] - floor( CMx[fullerene] / boxMax.x) * boxMax.x )/DLp.x); 	
 	  	posy = (int)(( CMy[fullerene] - floor( CMy[fullerene] / boxMax.y) * boxMax.y )/DLp.y); 	
	  	posz = (int)(( CMz[fullerene] - floor( CMz[fullerene] / boxMax.z) * boxMax.z )/DLp.z); 		
	 
		int j1 = 0;
	  	int j2 = 0;
	  	int j3 = 0;
	 
	  	for (  int i = -1; i < 2 ; ++i ){
				
			j1 = posx + i;
			if (j1 >= Xdiv || j1 <= -1) continue; 
				 

			for (  int j = -1; j < 2; ++j ){
					
				j2 = posy + j;
				j2 = j2 - floor((float)j2/(float)Ydiv) * Ydiv;	 
	
				for (  int k = -1 ; k < 2; ++k ){
			
			
					j3 = posz + k;
					
					if(useRigidBoxZ){
					
						if(j3 < 0 || j3 > Zdiv-1) continue;
					
					}else{	
					
						j3 = j3 - floor((float)j3/(float)Zdiv) * Zdiv;	 
					
					}

			  		int index = atomicAdd( &d_NoofNNlist[j3*Xdiv*Ydiv+j2*Xdiv+j1] , 1); //returns old
#ifdef PRINT_TOO_SHORT_ERROR
			  		if ( index > 64 )
					{
                         			printf("Fullerene %d, NN-list too short, atleast %d\n", fullerene, index);
                                  		// for ( int k = 0; k < 32; ++k )
                                  		// printf("%d ",d_NNlist[ 32*(j2*Xdiv+j1) + k]);
                                  		// printf("\n");
						continue;
					}
#endif
			  		d_NNlist[ 64*(j3*Xdiv*Ydiv+j2*Xdiv+j1)+index] = fullerene;
					
				}
	
			}
		}
		
				
		if(posx == Xdiv-1 ){
			
			posy = (int)(( (CMy[fullerene] - Pshift) - floor((CMy[fullerene] - Pshift) / boxMax.y) * boxMax.y )/DLp.y);	
			
			j1 = 0;
			j3 = posz;
			if(useRigidBoxZ){
				if ( posz < 0 ) posz = 0;
	  			if ( posz > Zdiv - 1 ) posz = Zdiv - 1;	
			}else{						
				j3 = j3 - floor((float)j3/(float)Zdiv) * Zdiv;	 
			}
				 
			for (  int i = -1; i < 2 ; ++i ){
				
				j2 = posy + i;
				j2 = j2 - floor((float)j2/(float)Ydiv) * Ydiv;
		
				int index = atomicAdd( &d_NoofNNlist[j3*Xdiv*Ydiv+j2*Xdiv+j1] , 1);
				d_NNlist[ 64*(j3*Xdiv*Ydiv+j2*Xdiv+j1)+index] = fullerene;
		
			}
		}
			
			
			
			
		if (posx == 0){
			
			posy = (int)(( (CMy[fullerene] + Pshift) - floor( (CMy[fullerene] + Pshift) / boxMax.y) * boxMax.y )/DLp.y);	
			
			j1 = Xdiv - 1;
			j3 = posz;
			if(useRigidBoxZ){
				if ( posz < 0 ) posz = 0;
	  			if ( posz > Zdiv - 1 ) posz = Zdiv - 1;	
			}else{						
				j3 = j3 - floor((float)j3/(float)Zdiv) * Zdiv;	 
			}
				
			for (  int i = -1; i < 2 ; ++i ){
				
				j2 = posy + i;
				j2 = j2 - floor((float)j2/(float)Ydiv) * Ydiv;
		
				int index = atomicAdd( &d_NoofNNlist[j3*Xdiv*Ydiv+j2*Xdiv+j1] , 1);
				d_NNlist[ 64*(j3*Xdiv*Ydiv+j2*Xdiv+j1)+index] = fullerene;
		
			}	
			
				

		}
			
		
		CMxNNlist[fullerene] = CMx[fullerene];
		CMyNNlist[fullerene] = CMy[fullerene];
		CMzNNlist[fullerene] = CMz[fullerene];
		
		
	}
		
}


// Pinning
__global__ void makeNNlistPin(int impurityNum, float *CMx, float *CMy,float *CMz,
                           int Xdiv, int Ydiv, int Zdiv, float3 BoxMin,
                           int *d_NoofNNlistPin, int *d_NNlistPin, float DL)
{


	int fullerene = blockIdx.x*blockDim.x+threadIdx.x;


	if ( fullerene < impurityNum )
	{
	  
		int posx = 0;
		int posy = 0;
		int posz = 0;		


	 	posx = (int)((CMx[fullerene] - BoxMin.x)/DL);
	  	if ( posx < 0 ) posx = 0;
	  	if ( posx > Xdiv - 1 ) posx = Xdiv - 1;
	  	

	  	posy = (int)((CMy[fullerene]-BoxMin.y)/DL);
	  	if ( posy < 0 ) posy = 0;
	  	if ( posy > Ydiv - 1 ) posy = Ydiv - 1;

	   	posz = (int)((CMz[fullerene]-BoxMin.z)/DL);
	  	if ( posz < 0 ) posz = 0;
	  	if ( posz > Zdiv - 1 ) posz = Zdiv - 1;
	  	
	 
		int j1 = 0;
	  	int j2 = 0;
	  	int j3 = 0;
	 
	  	for (  int i = -1; i < 2 ; ++i ){
				
			j1 = posx + i;
			if(j1 < 0 || j1 > Xdiv-1) continue;
			

			for (  int j = -1; j < 2; ++j ){
					
				j2 = posy + j;
				if(j2 < 0 || j2 > Ydiv-1) continue;
				
	
				for (  int k = -1 ; k < 2; ++k ){
			
					j3 = posz + k;
					if(j3 < 0 || j3 > Zdiv-1) continue;
		

			  		int index = atomicAdd( &d_NoofNNlistPin[j3*Xdiv*Ydiv+j2*Xdiv+j1] , 1); //returns old
#ifdef PRINT_TOO_SHORT_ERROR
			  		if ( index > 32 )
					{
                         			printf("Fullerene %d, NN-listPin too short, atleast %d\n", fullerene, index);
                                  			// for ( int k = 0; k < 32; ++k )
                                  			//     printf("%d ",d_NNlist[ 32*(j2*Xdiv+j1) + k]); 
                                 			// printf("\n");
						 continue;
					}
#endif
			  		d_NNlistPin[ 32*(j3*Xdiv*Ydiv+j2*Xdiv+j1)+index] = fullerene;
					
				}
	
			}
		}	
		
	}

}




__global__ void makeNNlistPBCPin(int impurityNum, float *CMx, float *CMy,float *CMz,
                           float attrac, int Xdiv, int Ydiv, int Zdiv, float3 boxMax,
                           int *d_NoofNNlistPin, int *d_NNlistPin, float3 DLp, bool useRigidBoxZ, bool useRigidBoxY)
{


	int fullerene = blockIdx.x*blockDim.x+threadIdx.x;
//  printf("(%d, %d, %d) %d %d\n", blockIdx.x, blockDim.x, threadIdx.x, fullerene, No_of_C180s);


	if ( fullerene < impurityNum )
	{
	  
		int posx = 0;
		int posy = 0;
		int posz = 0;	
		float boxX = boxMax.x;
		float boxY = boxMax.y;
		float boxZ = boxMax.z;	
		float DX = DLp.x;
		float DY = DLp.y;
		float DZ = DLp.z;	

	  	
	  	posx = (int)(( CMx[fullerene] - floor( CMx[fullerene] / boxX) * boxX )/DX); 	
 	  	posy = (int)(( CMy[fullerene] - floor( CMy[fullerene] / boxY) * boxY )/DY); 	
	  	posz = (int)(( CMz[fullerene] - floor( CMz[fullerene] / boxZ) * boxZ )/DZ); 		
	 
		int j1 = 0;
	  	int j2 = 0;
	  	int j3 = 0;
	 
	  	for (  int i = -1; i < 2 ; ++i ){
				
			j1 = posx + i;
			j1 = j1 - floor((float)j1/(float)Xdiv) * Xdiv;	 

			for (  int j = -1; j < 2; ++j ){ 


				j2 = posy + j;
					
				if(useRigidBoxY){
					
					if(j2 < 0 || j2 > Ydiv-1) continue;
					
				}else{	
					
					j2 = j2 - floor((float)j2/(float)Ydiv) * Ydiv;	 
					
				}
	
				for (  int k = -1 ; k < 2; ++k ){
			
			
					j3 = posz + k;
					
					if(useRigidBoxZ){
					
						if(j3 < 0 || j3 > Zdiv-1) continue;
					
					}else{	
					
						j3 = j3 - floor((float)j3/(float)Zdiv) * Zdiv;	 
					
					}

			  		int index = atomicAdd( &d_NoofNNlistPin[j3*Xdiv*Ydiv+j2*Xdiv+j1] , 1); //returns old
#ifdef PRINT_TOO_SHORT_ERROR
			  		if ( index > 32 )
					{
                         			printf("Fullerene %d, NN-listPin too short, atleast %d\n", fullerene, index);
                                  		// for ( int k = 0; k < 32; ++k )
                                  		// printf("%d ",d_NNlist[ 32*(j2*Xdiv+j1) + k]);
                                  		// printf("\n");
						continue;
					}
#endif
			  		d_NNlistPin[ 32*(j3*Xdiv*Ydiv+j2*Xdiv+j1)+index] = fullerene;
					
				}
	
			}
		}	

		
		
	}
		

}



__global__ void makeNNlistLEbcPin(int impurityNum, float *CMx, float *CMy,float *CMz,
                           float attrac, int Xdiv, int Ydiv, int Zdiv, float3 boxMax,
                           int *d_NoofNNlistPin, int *d_NNlistPin, float3 DLp, float Pshift,bool useRigidBoxZ)
{


	int fullerene = blockIdx.x*blockDim.x+threadIdx.x;


	if ( fullerene < impurityNum )
	{
	  
		int posx = 0;
		int posy = 0;
		int posz = 0;		

	  	
	  	posx = (int)(( CMx[fullerene] - floor( CMx[fullerene] / boxMax.x) * boxMax.x )/DLp.x); 	
 	  	posy = (int)(( CMy[fullerene] - floor( CMy[fullerene] / boxMax.y) * boxMax.y )/DLp.y); 	
	  	posz = (int)(( CMz[fullerene] - floor( CMz[fullerene] / boxMax.z) * boxMax.z )/DLp.z); 		
	 
		int j1 = 0;
	  	int j2 = 0;
	  	int j3 = 0;
	 
	  	for (  int i = -1; i < 2 ; ++i ){
				
			j1 = posx + i;
			if (j1 >= Xdiv || j1 <= -1) continue; 
				 

			for (  int j = -1; j < 2; ++j ){
					
				j2 = posy + j;
				j2 = j2 - floor((float)j2/(float)Ydiv) * Ydiv;	 
	
				for (  int k = -1 ; k < 2; ++k ){
			
			
					j3 = posz + k;
					
					if(useRigidBoxZ){
					
						if(j3 < 0 || j3 > Zdiv-1) continue;
					
					}else{	
					
						j3 = j3 - floor((float)j3/(float)Zdiv) * Zdiv;	 
					
					}

			  		int index = atomicAdd( &d_NoofNNlistPin[j3*Xdiv*Ydiv+j2*Xdiv+j1] , 1); //returns old
#ifdef PRINT_TOO_SHORT_ERROR
			  		if ( index > 32 )
					{
                         			printf("Fullerene %d, NN-listPin too short, atleast %d\n", fullerene, index);
                                  		// for ( int k = 0; k < 32; ++k )
                                  		// printf("%d ",d_NNlist[ 32*(j2*Xdiv+j1) + k]);
                                  		// printf("\n");
						continue;
					}
#endif
			  		
			  		d_NNlistPin[ 32*(j3*Xdiv*Ydiv+j2*Xdiv+j1)+index] = fullerene;
					
				}
	
			}
		}
		
				
		if(posx == Xdiv-1 ){
			
			posy = (int)(( (CMy[fullerene] - Pshift) - floor((CMy[fullerene] - Pshift) / boxMax.y) * boxMax.y )/DLp.y);	
			
			j1 = 0;
			j3 = posz;
			if(useRigidBoxZ){
				if ( posz < 0 ) posz = 0;
	  			if ( posz > Zdiv - 1 ) posz = Zdiv - 1;	
			}else{						
				j3 = j3 - floor((float)j3/(float)Zdiv) * Zdiv;	 
			}
				 
			for (  int i = -1; i < 2 ; ++i ){
				
				j2 = posy + i;
				j2 = j2 - floor((float)j2/(float)Ydiv) * Ydiv;
		
				int index = atomicAdd( &d_NoofNNlistPin[j3*Xdiv*Ydiv+j2*Xdiv+j1] , 1);
				d_NNlistPin[ 32*(j3*Xdiv*Ydiv+j2*Xdiv+j1)+index] = fullerene;
		
			}
		}
			
			
			
			
		if (posx == 0){
			
			posy = (int)(( (CMy[fullerene] + Pshift) - floor( (CMy[fullerene] + Pshift) / boxMax.y) * boxMax.y )/DLp.y);	
			
			j1 = Xdiv - 1;
			j3 = posz;
			if(useRigidBoxZ){
				if ( posz < 0 ) posz = 0;
	  			if ( posz > Zdiv - 1 ) posz = Zdiv - 1;	
			}else{						
				j3 = j3 - floor((float)j3/(float)Zdiv) * Zdiv;	 
			}
				
			for (  int i = -1; i < 2 ; ++i ){
				
				j2 = posy + i;
				j2 = j2 - floor((float)j2/(float)Ydiv) * Ydiv;
		
				int index = atomicAdd( &d_NoofNNlistPin[j3*Xdiv*Ydiv+j2*Xdiv+j1] , 1);
				d_NNlistPin[ 32*(j3*Xdiv*Ydiv+j2*Xdiv+j1)+index] = fullerene;
		
			}	
			
				

		}
			

		
		
	}
		

}


__global__ void DangerousParticlesFinder(int No_of_C180s, float *CMx, float *CMy,float *CMz,
					  float *CMxNNlist, float *CMyNNlist, float *CMzNNlist,
					  float BufferDistance, int *num_cell_dang, int* cell_dang_inds, char* cell_dang,
					  float3 boxMax)
{


	int fullerene = blockIdx.x*blockDim.x+threadIdx.x;


	if ( fullerene < No_of_C180s )
	{
		
		if( cell_dang[fullerene] == 0){
			
			float deltaX, deltaY, deltaZ;
			float R;
		
		
			deltaX = CMxNNlist[fullerene] - CMx[fullerene]; 
			deltaY = CMyNNlist[fullerene] - CMy[fullerene];
			deltaZ = CMzNNlist[fullerene] - CMz[fullerene];
		
			R  = deltaX*deltaX+deltaY*deltaY+deltaZ*deltaZ;
		
			if (R >= BufferDistance){
			
				cell_dang[fullerene] = 1;
				int index = atomicAdd(&num_cell_dang[0],1);   
				cell_dang_inds[index] = fullerene;   
		
			}
		}


	}

}


__global__ void DangerousParticlesFinderPBC(int No_of_C180s, float *CMx, float *CMy,float *CMz,
					  float *CMxNNlist, float *CMyNNlist, float *CMzNNlist,
					  float BufferDistance, int *num_cell_dang, int* cell_dang_inds, char* cell_dang,
					  float3 boxMax, bool useRigidBoxZ, bool useRigidBoxY)
{


	int fullerene = blockIdx.x*blockDim.x+threadIdx.x;


	if ( fullerene < No_of_C180s )
	{
		
		if( cell_dang[fullerene] == 0){
			
			float deltaX, deltaY, deltaZ;
			float R;
		
		
			deltaX = CMxNNlist[fullerene] - CMx[fullerene];
			deltaX = deltaX - nearbyint( deltaX / boxMax.x) * boxMax.x;
			 
			deltaY = CMyNNlist[fullerene] - CMy[fullerene];
			if (!useRigidBoxY)deltaY = deltaY - nearbyint( deltaY / boxMax.y) * boxMax.y;
			
			deltaZ = CMzNNlist[fullerene] - CMz[fullerene];
			if (!useRigidBoxZ) deltaZ = deltaZ - nearbyint( deltaZ / boxMax.z) * boxMax.z;
			
		
			R  = deltaX*deltaX+deltaY*deltaY+deltaZ*deltaZ;
		
			
			if (R >= BufferDistance){
			
				cell_dang[fullerene] = 1;
				int index = atomicAdd(&num_cell_dang[0],1);   
				cell_dang_inds[index] = fullerene; 
				//printf("cell %d\n",fullerene);  
		
			}
		}


	}

}


__global__ void DangerousParticlesFinderLEbc(int No_of_C180s, float *CMx, float *CMy,float *CMz,
					  float *CMxNNlist, float *CMyNNlist, float *CMzNNlist,
					  float BufferDistance, int *num_cell_dang, int* cell_dang_inds, char* cell_dang,
					  float3 boxMax, bool useRigidBoxZ, bool useRigidBoxY )
{


	int fullerene = blockIdx.x*blockDim.x+threadIdx.x;


	if ( fullerene < No_of_C180s )
	{
		
		if( cell_dang[fullerene] == 0){
			
			float deltaX, deltaY, deltaZ;
			float R;
		
		
			if ( abs(CMx[fullerene] - CMxNNlist[fullerene]) > boxMax.x/2 ){
				
				cell_dang[fullerene] = 1;
				int index = atomicAdd(&num_cell_dang[0],1);   
				cell_dang_inds[index] = fullerene; 
				
			
			} else{
		
			deltaX = CMxNNlist[fullerene] - CMx[fullerene];
			
			deltaY = CMyNNlist[fullerene] - CMy[fullerene];
			deltaY = deltaY - nearbyint( deltaY / boxMax.y) * boxMax.y;
			
			deltaZ = CMzNNlist[fullerene] - CMz[fullerene];
			if (!useRigidBoxZ) deltaZ = deltaZ - nearbyint( deltaZ / boxMax.z) * boxMax.z;
			 
						    			
            		R  = deltaX*deltaX+deltaY*deltaY+deltaZ*deltaZ;
        	
            		if (R >= BufferDistance){
					
					
				cell_dang[fullerene] = 1;
				int index = atomicAdd(&num_cell_dang[0],1);   
				cell_dang_inds[index] = fullerene;   
		
			}
            		
            		
            		}
			
			
		}


	}

}



__global__ void UpdateNNlistDivision(int No_of_C180s, int non_divided_cells, float *CMx, float *CMy,float *CMz, float *CMxNNlist, float *CMyNNlist, float *CMzNNlist,
                           		int Xdiv, int Ydiv, int Zdiv, float3 BoxMin,
                           		int *d_NoofNNlist, int *d_NNlist, float DL)
{


    	int fullerene  = non_divided_cells + blockIdx.x*blockDim.x+threadIdx.x;

	if ( fullerene < No_of_C180s )
	{  
		
		int posx = 0;
		int posy = 0;
		int posz = 0;		
		

		posx = (int)((CMx[fullerene] - BoxMin.x)/DL);
		if ( posx < 0 ) posx = 0;
		if ( posx > Xdiv - 1 ) posx = Xdiv - 1;
	  	

	 	posy = (int)((CMy[fullerene]-BoxMin.y)/DL);
	 	if ( posy < 0 ) posy = 0;
	 	if ( posy > Ydiv - 1 ) posy = Ydiv - 1;

	 	posz = (int)((CMz[fullerene]-BoxMin.z)/DL);
	 	if ( posz < 0 ) posz = 0;
	 	if ( posz > Zdiv - 1 ) posz = Zdiv - 1;
	  	
	 
	 	int j1 = 0;
	 	int j2 = 0;
	 	int j3 = 0;
	 
	 	for (  int i = -1; i < 2 ; ++i ){
				
			j1 = posx + i;
			if(j1 < 0 || j1 > Xdiv-1) continue;
			

			for (  int j = -1; j < 2; ++j ){
					
				j2 = posy + j;
				if(j2 < 0 || j2 > Ydiv-1) continue;
				
	
				for (  int k = -1 ; k < 2; ++k ){
			
					j3 = posz + k;
					if(j3 < 0 || j3 > Zdiv-1) continue;
		

			  		int index = atomicAdd( &d_NoofNNlist[j3*Xdiv*Ydiv+j2*Xdiv+j1] , 1); //returns old
#ifdef PRINT_TOO_SHORT_ERROR
			  		if ( index > 64 )
					{
					        printf("Fullerene %d, NN-list too short, atleast %d\n", fullerene, index);
               	                   	// for ( int k = 0; k < 32; ++k )
               	                   	//     printf("%d ",d_NNlist[ 32*(j2*Xdiv+j1) + k]); 
               	                  	// printf("\n");
						 continue;
					}
#endif
				  	d_NNlist[ 64*(j3*Xdiv*Ydiv+j2*Xdiv+j1)+index] = fullerene;
					
				}
	
			}
		}	
		
	
		
		CMxNNlist[fullerene] = CMx[fullerene];
		CMyNNlist[fullerene] = CMy[fullerene];
		CMzNNlist[fullerene] = CMz[fullerene];
	
	}

}

__global__ void UpdateNNlistDivisionPBC(int No_of_C180s, int non_divided_cells, float *CMx, float *CMy,float *CMz, float *CMxNNlist, float *CMyNNlist, float *CMzNNlist,
                           		 int Xdiv, int Ydiv, int Zdiv, float3 boxMax,
                           		int *d_NoofNNlist, int *d_NNlist, float3 DLp, bool useRigidBoxZ, bool useRigidBoxY)
{


    	int fullerene  = non_divided_cells + blockIdx.x*blockDim.x+threadIdx.x;
    	
	if ( fullerene < No_of_C180s )
	{ 
		int posx = 0;
		int posy = 0;
		int posz = 0;		

	  	
		posx = (int)(( CMx[fullerene] - floor( CMx[fullerene] / boxMax.x) * boxMax.x )/DLp.x); 	
 		posy = (int)(( CMy[fullerene] - floor( CMy[fullerene] / boxMax.y) * boxMax.y )/DLp.y); 	
		posz = (int)(( CMz[fullerene] - floor( CMz[fullerene] / boxMax.z) * boxMax.z )/DLp.z); 		
	 
		int j1 = 0;
		int j2 = 0;
		int j3 = 0;
	 
		for (  int i = -1; i < 2 ; ++i ){
				
			j1 = posx + i;
			j1 = j1 - floor((float)j1/(float)Xdiv) * Xdiv;	 

			for (  int j = -1; j < 2; ++j ){ 


				j2 = posy + j;
						
				if(useRigidBoxY){
						
					if(j2 < 0 || j2 > Ydiv-1) continue;
					
				}else{	
					
					j2 = j2 - floor((float)j2/(float)Ydiv) * Ydiv;	 
					
				}
	
				for (  int k = -1 ; k < 2; ++k ){
			
			
					j3 = posz + k;
						
					if(useRigidBoxZ){
					
						if(j3 < 0 || j3 > Zdiv-1) continue;
					
					}else{	
					
						j3 = j3 - floor((float)j3/(float)Zdiv) * Zdiv;	 
					
					}

				  	int index = atomicAdd( &d_NoofNNlist[j3*Xdiv*Ydiv+j2*Xdiv+j1] , 1); //returns old
#ifdef PRINT_TOO_SHORT_ERROR
				  	if ( index > 64 )
					{
               	          		printf("Fullerene %d, NN-list too short, atleast %d\n", fullerene, index);
               	                   	// for ( int k = 0; k < 32; ++k )
               	                   	// printf("%d ",d_NNlist[ 32*(j2*Xdiv+j1) + k]);
               	                   	// printf("\n");
						continue;
					}
#endif
				  	d_NNlist[ 64*(j3*Xdiv*Ydiv+j2*Xdiv+j1)+index] = fullerene;
					
				}
	
			}
		}	

		
		CMxNNlist[fullerene] = CMx[fullerene];
		CMyNNlist[fullerene] = CMy[fullerene];
		CMzNNlist[fullerene] = CMz[fullerene];
	
	}
		
}

__global__ void UpdateNNlistDivisionLEbc(int No_of_C180s, int non_divided_cells, float *CMx, float *CMy,float *CMz, float *CMxNNlist, float *CMyNNlist, float *CMzNNlist,
                           		   int Xdiv, int Ydiv, int Zdiv, float3 boxMax,
                           		   int *d_NoofNNlist, int *d_NNlist, float3 DLp, float Pshift,bool useRigidBoxZ)
{


    	int fullerene  = non_divided_cells + blockIdx.x*blockDim.x+threadIdx.x;
	
	if ( fullerene < No_of_C180s )
	{ 	  
		int posx = 0;
		int posy = 0;
		int posz = 0;		

	  	
	  	posx = (int)(( CMx[fullerene] - floor( CMx[fullerene] / boxMax.x) * boxMax.x )/DLp.x); 	
 	  	posy = (int)(( CMy[fullerene] - floor( CMy[fullerene] / boxMax.y) * boxMax.y )/DLp.y); 	
	  	posz = (int)(( CMz[fullerene] - floor( CMz[fullerene] / boxMax.z) * boxMax.z )/DLp.z); 		
	 
		int j1 = 0;
	  	int j2 = 0;
	  	int j3 = 0;
	 
	  	for (  int i = -1; i < 2 ; ++i ){
				
			j1 = posx + i;
			if (j1 >= Xdiv || j1 <= -1) continue; 
				 

			for (  int j = -1; j < 2; ++j ){
					
				j2 = posy + j;
				j2 = j2 - floor((float)j2/(float)Ydiv) * Ydiv;	 
	
				for (  int k = -1 ; k < 2; ++k ){
			
			
					j3 = posz + k;
					
					if(useRigidBoxZ){
					
						if(j3 < 0 || j3 > Zdiv-1) continue;
					
					}else{	
					
						j3 = j3 - floor((float)j3/(float)Zdiv) * Zdiv;	 
					
					}

			  		int index = atomicAdd( &d_NoofNNlist[j3*Xdiv*Ydiv+j2*Xdiv+j1] , 1); //returns old
#ifdef PRINT_TOO_SHORT_ERROR
			  		if ( index > 64 )
					{
                         			printf("Fullerene %d, NN-list too short, atleast %d\n", fullerene, index);
                                  		// for ( int k = 0; k < 32; ++k )
                                  		// printf("%d ",d_NNlist[ 32*(j2*Xdiv+j1) + k]);
                                  		// printf("\n");
						continue;
					}
#endif
			  		d_NNlist[ 64*(j3*Xdiv*Ydiv+j2*Xdiv+j1)+index] = fullerene;
					
				}
	
			}
		}
		
				
		if(posx == Xdiv-1 ){
			
			posy = (int)(( (CMy[fullerene] - Pshift) - floor((CMy[fullerene] - Pshift) / boxMax.y) * boxMax.y )/DLp.y);	
			
			j1 = 0;
			j3 = posz;
			if(useRigidBoxZ){
				if ( posz < 0 ) posz = 0;
	  			if ( posz > Zdiv - 1 ) posz = Zdiv - 1;	
			}else{						
				j3 = j3 - floor((float)j3/(float)Zdiv) * Zdiv;	 
			}
				 
			for (  int i = -1; i < 2 ; ++i ){
				
				j2 = posy + i;
				j2 = j2 - floor((float)j2/(float)Ydiv) * Ydiv;
		
				int index = atomicAdd( &d_NoofNNlist[j3*Xdiv*Ydiv+j2*Xdiv+j1] , 1);
				d_NNlist[ 64*(j3*Xdiv*Ydiv+j2*Xdiv+j1)+index] = fullerene;
		
			}
		}
			
						
		if (posx == 0){
			
			posy = (int)(( (CMy[fullerene] + Pshift) - floor( (CMy[fullerene] + Pshift) / boxMax.y) * boxMax.y )/DLp.y);	
			
			j1 = Xdiv - 1;
			j3 = posz;
			if(useRigidBoxZ){
				if ( posz < 0 ) posz = 0;
	  			if ( posz > Zdiv - 1 ) posz = Zdiv - 1;	
			}else{						
				j3 = j3 - floor((float)j3/(float)Zdiv) * Zdiv;	 
			}
				
			for (  int i = -1; i < 2 ; ++i ){
				
				j2 = posy + i;
				j2 = j2 - floor((float)j2/(float)Ydiv) * Ydiv;
		
				int index = atomicAdd( &d_NoofNNlist[j3*Xdiv*Ydiv+j2*Xdiv+j1] , 1);
				d_NNlist[ 64*(j3*Xdiv*Ydiv+j2*Xdiv+j1)+index] = fullerene;
		
			}	
			
				

		}
			
		
		CMxNNlist[fullerene] = CMx[fullerene];
		CMyNNlist[fullerene] = CMy[fullerene];
		CMzNNlist[fullerene] = CMz[fullerene];
	}	
}



__global__ void Ghost_Cells_Pack(int No_of_Ghost_cells_buffer, int* d_Ghost_Cells_ind,
				float *d_X,  float *d_Y,  float *d_Z,
                               float* d_velListX, float* d_velListY, float* d_velListZ,
                               float* d_CMx, float* d_CMy, float* d_CMz,
				float *d_X_gc_buffer,  float *d_Y_gc_buffer,  float *d_Z_gc_buffer,
                              float* d_velListX_gc_buffer, float* d_velListY_gc_buffer, float* d_velListZ_gc_buffer,
                              float* d_CMx_gc_buffer, float* d_CMy_gc_buffer, float* d_CMz_gc_buffer){

	
	
	int ghost_cell = d_Ghost_Cells_ind[blockIdx.x];
	
	int tid = threadIdx.x;	
	int cell = blockIdx.x;	
	
	if( cell < No_of_Ghost_cells_buffer ) {
		

		d_X_gc_buffer[cell*192 + tid] = d_X[192*ghost_cell + tid];
		d_Y_gc_buffer[cell*192 + tid] = d_Y[192*ghost_cell + tid];
		d_Z_gc_buffer[cell*192 + tid] = d_Z[192*ghost_cell + tid];
	
	
		d_velListX_gc_buffer[cell*192 + tid] = d_velListX[192*ghost_cell + tid];
		d_velListY_gc_buffer[cell*192 + tid] = d_velListY[192*ghost_cell + tid];
		d_velListZ_gc_buffer[cell*192 + tid] = d_velListZ[192*ghost_cell + tid]; 
		
		
		if(tid == 0){
			
			d_CMx_gc_buffer[cell] = d_CMx[ghost_cell];
			d_CMy_gc_buffer[cell] = d_CMy[ghost_cell];
			d_CMz_gc_buffer[cell] = d_CMz[ghost_cell];
		
		
		}

	}
	
	
	
}  

__global__ void UpdateNNlistWithGhostCells(int No_of_C180s, int All_Cells, float *d_CMx, float *d_CMy,float *d_CMz,
                           int Xdiv, int Ydiv, int Zdiv, float3 Subdivision_min,
                           int *d_NoofNNlist, int *d_NNlist, float DL){
                           
	
	int atom = blockIdx.x*blockDim.x+threadIdx.x;
	

	
	if ( atom < All_Cells )
	{
	
		int fullerene = atom + No_of_C180s;	
		//printf("fullerene:	%d\n",fullerene);
		  
		int posx = 0;
		int posy = 0;
		int posz = 0;		
		

	 	posx = (int)((d_CMx[fullerene] - Subdivision_min.x)/DL);
	  	if ( posx < 0 ) posx = 0;
	  	if ( posx > Xdiv - 1 ) posx = Xdiv - 1;
	  	

	  	posy = (int)((d_CMy[fullerene] - Subdivision_min.y)/DL);
	  	if ( posy < 0 ) posy = 0;
	  	if ( posy > Ydiv - 1 ) posy = Ydiv - 1;

	   	posz = (int)((d_CMz[fullerene] - Subdivision_min.z)/DL);
	  	if ( posz < 0 ) posz = 0;
	  	if ( posz > Zdiv - 1 ) posz = Zdiv - 1;

		
		int j1 = 0;
		int j2 = 0;
		int j3 = 0;
		
		
		//printf(" my min x is %f, and posx:	%d, posy:	%d, posz:	%d\n",Subdivision_min.x, posx,posy,posz);
		
		for (  int i = -1; i < 2 ; ++i ){
				
			j1 = posx + i;
			if (j1 >= Xdiv || j1 <= -1) continue; 

			for (  int j = -1; j < 2; ++j ){
					
				j2 = posy + j;
				if(j2 < 0 || j2 > Ydiv-1) continue;
				
	
				for (  int k = -1 ; k < 2; ++k ){
			
					j3 = posz + k;
					if(j3 < 0 || j3 > Zdiv-1) continue;
		

					int index = atomicAdd( &d_NoofNNlist[j3*Xdiv*Ydiv+j2*Xdiv+j1] , 1); //returns old
			  		
#ifdef PRINT_TOO_SHORT_ERROR
					if ( index > 64 )
					{
                				printf("Fullerene %d, NN-list too short, atleast %d\n", fullerene, index);
                      				 // for ( int k = 0; k < 32; ++k )
                      				 //     printf("%d ",d_NNlist[ 32*(j2*Xdiv+j1) + k]); 
                      				 // printf("\n");
						continue;
					}
#endif
					d_NNlist[ 64*(j3*Xdiv*Ydiv+j2*Xdiv+j1)+index] = fullerene;
				}
			}
		}
	}
                           
}



__global__ void migrated_cells_finder(int No_of_C180s, float *d_CM,
                         		float Sub_max, float Sub_min, float BMin, float BMax,
                         		int* d_counter_mc_r, int* d_counter_mc_l,
                         		int* d_migrated_cells_ind_R, int* d_migrated_cells_ind_L,
                         		char* d_cell_mig){
 
	
	int fullerene = blockIdx.x*blockDim.x+threadIdx.x;


	if ( fullerene < No_of_C180s )
	{
	      
      		float C = d_CM[fullerene];
      		
      			
      		if ( BMin < Sub_min) {	
      			
      			if ( C < Sub_min - 0.1 ) {
	
	  			int index = atomicAdd(d_counter_mc_l,1);
	 			d_migrated_cells_ind_L[index] = fullerene;
	 			d_cell_mig[fullerene] = 1;
	  		
	  		}
	  	
	  	}		  	
	  	if (Sub_max < BMax){
	  		
	  		if ( C > Sub_max + 0.1) {
	  		
	  			int index = atomicAdd(d_counter_mc_r,1);
	 			d_migrated_cells_ind_R[index] = fullerene;
	 			d_cell_mig[fullerene] = 1;
	  		
	  		}
	  	}
		          		 
	}
                       		 
}



__global__ void ghost_cells_finder_Auxiliary(int No_of_C180s, int All_Cells, float *d_CM , 
						float Sub_max, float Sub_min,
						int* d_counter_gc_r, int* d_counter_gc_l,
                         			int* d_Ghost_Cells_ind_R, int* d_Ghost_Cells_ind_L)
{
 
	int atom = blockIdx.x*blockDim.x+threadIdx.x;
	
	if ( atom < All_Cells )
	{	
		int fullerene = atom + No_of_C180s;
	 	
	 	float pos = d_CM[fullerene];
	 	
	 	if( pos <=  Sub_min + 2.0 ){
	 			
	 		int index = atomicAdd(d_counter_gc_l,1);
	 		d_Ghost_Cells_ind_L[index] = fullerene;
	 	
	 	} else if( pos >=  Sub_max - 2.0 ){
	 			
	 		int index = atomicAdd(d_counter_gc_r,1);
	 		d_Ghost_Cells_ind_R[index] = fullerene;	 			
	 	}
	 	
	
	}
 
                         		 
}


__global__ void migrated_Cells_Remove_Pack(int No_of_C180s, int No_of_migration_cells_buffer, int* d_counter,
					 	int* d_migrated_cells_ind, char* d_cell_mig,
   						float *d_X,  float *d_Y,  float *d_Z,
                               		float* d_velListX, float* d_velListY, float* d_velListZ,
                               		float* d_CMx, float* d_CMy, float* d_CMz,
                               		float* d_ScaleFactor,float* d_Youngs_mod, float* d_Growth_rate, float* d_DivisionVolume,
                               		float* d_gamma_env, float* d_viscotic_damp, float* d_pressList, int* d_CellINdex, 
                               		float* d_Apo_rate, float* d_squeeze_rate,
						float *d_X_mc_buffer,  float *d_Y_mc_buffer,  float *d_Z_mc_buffer,
                               		float* d_velListX_mc_buffer, float* d_velListY_mc_buffer, float* d_velListZ_mc_buffer,
                               		float* d_CMx_mc_buffer, float* d_CMy_mc_buffer, float* d_CMz_mc_buffer,
                               		float* d_ScaleFactor_mc_buffer,float* d_Youngs_mod_mc_buffer, float* d_Growth_rate_mc_buffer, float* d_DivisionVolume_mc_buffer,
                               		float* d_gamma_env_mc_buffer, float* d_viscotic_damp_mc_buffer, float* d_pressList_mc_buffer, int* d_CellINdex_mc_buffer, 
                               		float* d_Apo_rate_mc_buffer, float* d_squeeze_rate_mc_buffer){

	
	
	int migrated_cell = d_migrated_cells_ind[blockIdx.x];
	
	int tid = threadIdx.x;	
	int cell = blockIdx.x;	
	
	if( cell < No_of_migration_cells_buffer ) {

		if( migrated_cell < No_of_C180s - No_of_migration_cells_buffer ){
		
			
			__shared__ int moving_Cell;
		
			if (tid == 0){

				int index = atomicAdd(d_counter,1);
				moving_Cell = No_of_C180s - index - 1;			
		
				while ( d_cell_mig[moving_Cell] == 1 ){
				
					index = atomicAdd(d_counter,1);
					moving_Cell = No_of_C180s - index - 1;
				}
	
			}
	
			
			__syncthreads();
			
			
			
			d_X_mc_buffer[cell*192 + tid] = d_X[192*migrated_cell + tid];
			d_X[migrated_cell*192 + tid] = d_X[192*moving_Cell + tid];
			
			d_Y_mc_buffer[cell*192 + tid] = d_Y[192*migrated_cell + tid];
			d_Y[migrated_cell*192 + tid] = d_Y[192*moving_Cell + tid];
			
			
			d_Z_mc_buffer[cell*192 + tid] = d_Z[192*migrated_cell + tid];
			d_Z[migrated_cell*192 + tid] = d_Z[192*moving_Cell + tid];
			

			d_velListX_mc_buffer[cell*192 + tid] = d_velListX[192*migrated_cell + tid];
			d_velListX[migrated_cell*192 + tid] = d_velListX[192*moving_Cell + tid];
			
			d_velListY_mc_buffer[cell*192 + tid] = d_velListY[192*migrated_cell + tid];
			d_velListY[migrated_cell*192 + tid] = d_velListY[192*moving_Cell + tid];
			
			
			d_velListZ_mc_buffer[cell*192 + tid] = d_velListZ[192*migrated_cell + tid]; 
			d_velListZ[migrated_cell*192 + tid] = d_velListZ[192*moving_Cell + tid];
			
			
			
			if(tid == 0){
			
				d_CMx_mc_buffer[cell] = d_CMx[migrated_cell];
				d_CMx[migrated_cell] = d_CMx[moving_Cell];
				
				d_CMy_mc_buffer[cell] = d_CMy[migrated_cell];
				d_CMy[migrated_cell] = d_CMy[moving_Cell];
				
				d_CMz_mc_buffer[cell] = d_CMz[migrated_cell];
				d_CMz[migrated_cell] = d_CMz[moving_Cell];
				
				d_pressList_mc_buffer[cell] = d_pressList[migrated_cell];
				d_pressList[migrated_cell] = d_pressList[moving_Cell];
				
				d_Growth_rate_mc_buffer[cell] = d_Growth_rate[migrated_cell];
				d_Growth_rate[migrated_cell] = d_Growth_rate[moving_Cell];
				
				d_Youngs_mod_mc_buffer[cell]  = d_Youngs_mod[migrated_cell];
				d_Youngs_mod[migrated_cell]  = d_Youngs_mod[moving_Cell];
				
				d_ScaleFactor_mc_buffer[cell] = d_ScaleFactor[migrated_cell];
				d_ScaleFactor[migrated_cell] = d_ScaleFactor[moving_Cell];
				
				d_DivisionVolume_mc_buffer[cell] = d_DivisionVolume[migrated_cell];
				d_DivisionVolume[migrated_cell] = d_DivisionVolume[moving_Cell];
				
				d_gamma_env_mc_buffer[cell] = d_gamma_env[migrated_cell];
				d_gamma_env[migrated_cell] = d_gamma_env[moving_Cell];
				
				d_viscotic_damp_mc_buffer[cell] = d_viscotic_damp[migrated_cell];
				d_viscotic_damp[migrated_cell] = d_viscotic_damp[moving_Cell];
				
				d_CellINdex_mc_buffer[cell] = d_CellINdex[migrated_cell];
				d_CellINdex[migrated_cell] = d_CellINdex[moving_Cell];
				
				d_Apo_rate_mc_buffer[cell] = d_Apo_rate[migrated_cell];
				d_Apo_rate[migrated_cell] = d_Apo_rate[moving_Cell];
				
				d_squeeze_rate_mc_buffer[cell] = d_squeeze_rate[migrated_cell];
				d_squeeze_rate[migrated_cell] = d_squeeze_rate[moving_Cell];
		
		
			}		
		
		
		
		
		
		} else {

			
			d_X_mc_buffer[cell*192 + tid] = d_X[192*migrated_cell + tid];
			d_Y_mc_buffer[cell*192 + tid] = d_Y[192*migrated_cell + tid];
			d_Z_mc_buffer[cell*192 + tid] = d_Z[192*migrated_cell + tid];
	
	
			d_velListX_mc_buffer[cell*192 + tid] = d_velListX[192*migrated_cell + tid];
			d_velListY_mc_buffer[cell*192 + tid] = d_velListY[192*migrated_cell + tid];
			d_velListZ_mc_buffer[cell*192 + tid] = d_velListZ[192*migrated_cell + tid]; 
		
		
			if(tid == 0){
			
				d_CMx_mc_buffer[cell] = d_CMx[migrated_cell];
				d_CMy_mc_buffer[cell] = d_CMy[migrated_cell];
				d_CMz_mc_buffer[cell] = d_CMz[migrated_cell];
				d_pressList_mc_buffer[cell] = d_pressList[migrated_cell];
				d_Growth_rate_mc_buffer[cell] = d_Growth_rate[migrated_cell];
				d_Youngs_mod_mc_buffer[cell]  = d_Youngs_mod[migrated_cell];
				d_ScaleFactor_mc_buffer[cell] = d_ScaleFactor[migrated_cell];
				d_DivisionVolume_mc_buffer[cell] = d_DivisionVolume[migrated_cell];
				d_gamma_env_mc_buffer[cell] = d_gamma_env[migrated_cell];
				d_viscotic_damp_mc_buffer[cell] = d_viscotic_damp[migrated_cell];
				d_CellINdex_mc_buffer[cell] = d_CellINdex[migrated_cell];
				d_Apo_rate_mc_buffer[cell] = d_Apo_rate[migrated_cell];
				d_squeeze_rate_mc_buffer[cell] = d_squeeze_rate[migrated_cell];
		
		
			}			


	
		}
	
	
	}
	
	
	
}  

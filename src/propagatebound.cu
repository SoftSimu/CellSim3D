#include <cuda.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "postscript.h"
#include "VectorFunctions.hpp"

#define PRINT_TOO_SHORT_ERROR


__global__ void makeNNlist(int No_of_C180s, float *CMx, float *CMy,float *CMz, float *CMxNNlist, float *CMyNNlist, float *CMzNNlist,
                           int Xdiv, int Ydiv, int Zdiv, double3 BoxMin,
                           int *d_NoofNNlist, int *d_NNlist, float DL, 
                           int MaxNeighList)
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
			  		if ( index > MaxNeighList )
					{
                         			printf("Fullerene %d, NN-list too short, atleast %d\n", fullerene, index);
                                  			// for ( int k = 0; k < 32; ++k )
                                  			//     printf("%d ",d_NNlist[ 32*(j2*Xdiv+j1) + k]); 
                                 			// printf("\n");
						 continue;
					}
#endif
			  		d_NNlist[ MaxNeighList*(j3*Xdiv*Ydiv+j2*Xdiv+j1)+index] = fullerene;
					
				}
	
			}
		}	
		
	
		
		CMxNNlist[fullerene] = CMx[fullerene];
		CMyNNlist[fullerene] = CMy[fullerene];
		CMzNNlist[fullerene] = CMz[fullerene];
	
	
	}

}

__global__ void makeNNlistMultiGpu( int No_of_C180s,  float R_ghost_buffer, float *d_CMx, float *d_CMy,float *d_CMz, float *CMxNNlist, float *CMyNNlist, float *CMzNNlist,
                           int Xdiv, int Ydiv, int Zdiv, float3 Subdivision_min, float3 Subdivision_max, double3 BoxMin, double3 boxMax,
                           int *d_NoofNNlist, int *d_NNlist, float DL, int* d_counter_gc_e, int* d_counter_gc_w,
                           int* d_counter_gc_n, int* d_counter_gc_s, int* d_counter_gc_u, int* d_counter_gc_d,
                           int* d_Ghost_Cells_ind_EAST, int* d_Ghost_Cells_ind_WEST, int* d_Ghost_Cells_ind_NORTH, int* d_Ghost_Cells_ind_SOUTH,
                           int* d_Ghost_Cells_ind_UP, int* d_Ghost_Cells_ind_DOWN,
                           int MaxNeighList )
{


	int fullerene = blockIdx.x*blockDim.x+threadIdx.x;
	//  printf("(%d, %d, %d) %d %d\n", blockIdx.x, blockDim.x, threadIdx.x, fullerene, No_of_C180s);


	if ( fullerene < No_of_C180s )
	{	
		
		
		float Cx = d_CMx[fullerene];
		float Cy = d_CMy[fullerene];
		float Cz = d_CMz[fullerene];
			
	 		
	 	if( Cx >  Subdivision_max.x - R_ghost_buffer){
	 			
	 		int index = atomicAdd(d_counter_gc_e,1);
	 		d_Ghost_Cells_ind_EAST[index] = fullerene;	 			
	 	}
	 	
	 	
	 	if( Cx <  Subdivision_min.x + R_ghost_buffer ){
	 			
	 		int index = atomicAdd(d_counter_gc_w,1);
	 		d_Ghost_Cells_ind_WEST[index] = fullerene;
	 		
	 	}
	 	
	 		
	 	if( Cy >  Subdivision_max.y - R_ghost_buffer ){
	 			
	 		int index = atomicAdd(d_counter_gc_n,1);
	 		d_Ghost_Cells_ind_NORTH[index] = fullerene;	 			
	 	}

	 		
	 	if( Cy <  Subdivision_min.y + R_ghost_buffer ){
	 			
	 		int index = atomicAdd(d_counter_gc_s,1);
	 		d_Ghost_Cells_ind_SOUTH[index] = fullerene;
	 	}

	 		
	 	if( Cz >  Subdivision_max.z - R_ghost_buffer ){
	 			
	 		int index = atomicAdd(d_counter_gc_u,1);
	 		d_Ghost_Cells_ind_UP[index] = fullerene;
	 				 			
	 	}


	 	if( Cz < Subdivision_min.z + R_ghost_buffer ){
	 			
	 		int index = atomicAdd(d_counter_gc_d,1);
	 		d_Ghost_Cells_ind_DOWN[index] = fullerene;
	 	}	

		
	 	int posx = (int)((Cx - Subdivision_min.x)/DL);
	  	if ( posx < 0 ) posx = 0;
	  	if ( posx > Xdiv - 1 ) posx = Xdiv - 1;
	  	

	  	int posy = (int)((Cy - Subdivision_min.y)/DL);
	  	if ( posy < 0 ) posy = 0;
	  	if ( posy > Ydiv - 1 ) posy = Ydiv - 1;

	   	int posz = (int)((Cz - Subdivision_min.z)/DL);
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
			  		if ( index > MaxNeighList )
					{
                         			printf("Fullerene %d, NN-list too short, atleast %d\n", fullerene, index);
                                  			// for ( int k = 0; k < 32; ++k )
                                  			//     printf("%d ",d_NNlist[ 32*(j2*Xdiv+j1) + k]); 
                                 			// printf("\n");
						 continue;
					}
#endif
			  		d_NNlist[MaxNeighList*(j3*Xdiv*Ydiv+j2*Xdiv+j1)+index] = fullerene;
					
				}
	
			}
		}	
			
		
		CMxNNlist[fullerene] = d_CMx[fullerene];
		CMyNNlist[fullerene] = d_CMy[fullerene];
		CMzNNlist[fullerene] = d_CMz[fullerene];
	
	
	}

}

__global__ void makeNNlistMultiGpuPBC( int No_of_C180s,  float R_ghost_buffer, float *d_CMx, float *d_CMy,float *d_CMz, float *CMxNNlist, float *CMyNNlist, float *CMzNNlist,
                           int Xdiv, int Ydiv, int Zdiv, float3 Subdivision_min, float3 Subdivision_max, double3 BoxMin, double3 boxMax,
                           int *d_NoofNNlist, int *d_NNlist, float DL, int* d_counter_gc_e, int* d_counter_gc_w,
                           int* d_counter_gc_n, int* d_counter_gc_s, int* d_counter_gc_u, int* d_counter_gc_d,
                           int* d_Ghost_Cells_ind_EAST, int* d_Ghost_Cells_ind_WEST, int* d_Ghost_Cells_ind_NORTH, int* d_Ghost_Cells_ind_SOUTH,
                           int* d_Ghost_Cells_ind_UP, int* d_Ghost_Cells_ind_DOWN,
                           int MaxNeighList)
{


	int fullerene = blockIdx.x*blockDim.x+threadIdx.x;


	if ( fullerene < No_of_C180s )
	{	
		
		float Cx = d_CMx[fullerene];
		float Cy = d_CMy[fullerene];
		float Cz = d_CMz[fullerene];

	 	
	 	if( Cx >  Subdivision_max.x - R_ghost_buffer){
	 			
	 		int index = atomicAdd(d_counter_gc_e,1);
	 		d_Ghost_Cells_ind_EAST[index] = fullerene;
	 			 			
	 	}
	 	
	 	
	 	if( Cx <  Subdivision_min.x + R_ghost_buffer ){
	 			
	 		int index = atomicAdd(d_counter_gc_w,1);
	 		d_Ghost_Cells_ind_WEST[index] = fullerene;
	 	}
	 	
	 		
	 	if( Cy >  Subdivision_max.y - R_ghost_buffer ){
	 			
	 		int index = atomicAdd(d_counter_gc_n,1);
	 		d_Ghost_Cells_ind_NORTH[index] = fullerene;	 			
	 	}
	 		
	 	if( Cy <  Subdivision_min.y + R_ghost_buffer ){
	 			
	 		int index = atomicAdd(d_counter_gc_s,1);
	 		d_Ghost_Cells_ind_SOUTH[index] = fullerene;
	 	}
	 		
	 	if( Cz >  Subdivision_max.z - R_ghost_buffer ){
	 			
	 		int index = atomicAdd(d_counter_gc_u,1);
	 		d_Ghost_Cells_ind_UP[index] = fullerene;
	 				 			
	 	}
	 
	 	if( Cz < Subdivision_min.z + R_ghost_buffer ){
	 			
	 		int index = atomicAdd(d_counter_gc_d,1);
	 		d_Ghost_Cells_ind_DOWN[index] = fullerene;
	 	}



	 	int posx = (int)((Cx - Subdivision_min.x)/DL);
	  	if ( posx < 0 ) posx = 0;
	  	else if ( posx > Xdiv - 1 ) posx = Xdiv - 1;
	  	

	  	int posy = (int)((Cy - Subdivision_min.y)/DL);
	  	if ( posy < 0 ) posy = 0;
	  	else if ( posy > Ydiv - 1 ) posy = Ydiv - 1;

	   	int posz = (int)((Cz - Subdivision_min.z)/DL);
	  	if ( posz < 0 ) posz = 0;
	  	else if ( posz > Zdiv - 1 ) posz = Zdiv - 1;
	 	
	 
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
			  		if ( index > MaxNeighList )
					{
                         			printf("Fullerene %d, NN-list too short, atleast %d\n", fullerene, index);
                                  			// for ( int k = 0; k < 32; ++k )
                                  			//     printf("%d ",d_NNlist[ 32*(j2*Xdiv+j1) + k]); 
                                 			// printf("\n");
						 continue;
					}
#endif
			  		d_NNlist[MaxNeighList*(j3*Xdiv*Ydiv+j2*Xdiv+j1)+index] = fullerene;
					
				}
	
			}
		}	
			
		
		CMxNNlist[fullerene] = d_CMx[fullerene];
		CMyNNlist[fullerene] = d_CMy[fullerene];
		CMzNNlist[fullerene] = d_CMz[fullerene];
	
	
	}

}


// Pinning
__global__ void makeNNlistPin(int impurityNum, float *CMx, float *CMy,float *CMz,
                           int Xdiv, int Ydiv, int Zdiv, float3 Subdivision_min,
                           int *d_NoofNNlistPin, int *d_NNlistPin, float DL)
{


	int fullerene = blockIdx.x*blockDim.x+threadIdx.x;


	if ( fullerene < impurityNum )
	{
	  
		int posx = 0;
		int posy = 0;
		int posz = 0;		


	 	posx = (int)((CMx[fullerene] - Subdivision_min.x)/DL);
	  	if ( posx < 0 ) posx = 0;
	  	if ( posx > Xdiv - 1 ) posx = Xdiv - 1;
	  	

	  	posy = (int)((CMy[fullerene] - Subdivision_min.y)/DL);
	  	if ( posy < 0 ) posy = 0;
	  	if ( posy > Ydiv - 1 ) posy = Ydiv - 1;

	   	posz = (int)((CMz[fullerene] - Subdivision_min.z)/DL);
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

__global__ void DangerousParticlesFinder(int No_of_C180s, float *CMx, float *CMy,float *CMz,
					  float *CMxNNlist, float *CMyNNlist, float *CMzNNlist,
					  float BufferDistance, int *d_num_cell_dang, int* cell_dang_inds, char* cell_dang,
					  int* d_num_cell_invalidator, float3 Subdivision_min, float DL)
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
				int index = atomicAdd(&d_num_cell_dang[0],1);   
				//cell_dang_inds[index] = fullerene;   
		
			}
		
		} else {
		
			
			int posOldx = (int)((CMxNNlist[fullerene] - Subdivision_min.x)/DL);
	  		int posNewx = (int)((CMx[fullerene] - Subdivision_min.x)/DL);
	  		
	  		int posOldy = (int)((CMyNNlist[fullerene] - Subdivision_min.y)/DL);
	  		int posNewy = (int)((CMy[fullerene] - Subdivision_min.y)/DL);
	  		
	   		int posOldz = (int)((CMzNNlist[fullerene] - Subdivision_min.z)/DL);
	   		int posNewz = (int)((CMzNNlist[fullerene] - Subdivision_min.z)/DL);
	   		
			if ( posOldx != posNewx ){
	  			
	  			int index = atomicAdd(&d_num_cell_invalidator[0],1);	
	  		
	  		}else if (posOldy != posNewy){
	  			
	  			int index = atomicAdd(&d_num_cell_invalidator[0],1);
	  		
	  		}else if (posOldz != posNewz){
	  		
	  			int index = atomicAdd(&d_num_cell_invalidator[0],1);
	  		}
		
		
		
		}


	}

}

__global__ void UpdateNNlistWithGhostCells(int No_of_C180s, int All_Cells, float *d_CMx, float *d_CMy,float *d_CMz,
                           		int Xdiv, int Ydiv, int Zdiv, float3 Subdivision_min,
                           		int *d_NoofNNlist, int *d_NNlist, float DL,
                           		int MaxNeighList){
                           
	
	int atom = blockIdx.x*blockDim.x+threadIdx.x;
	

	
	if ( atom < All_Cells )
	{
	
		int fullerene = atom + No_of_C180s;	
		

	 	int posx = (int)((d_CMx[fullerene] - Subdivision_min.x)/DL);
	  	if ( posx < 0 ) posx = 0;
	  	if ( posx > Xdiv - 1 ) posx = Xdiv - 1;
	  	

	  	int posy = (int)((d_CMy[fullerene] - Subdivision_min.y)/DL);
	  	if ( posy < 0 ) posy = 0;
	  	if ( posy > Ydiv - 1 ) posy = Ydiv - 1;

	   	int posz = (int)((d_CMz[fullerene] - Subdivision_min.z)/DL);
	  	if ( posz < 0 ) posz = 0;
	  	if ( posz > Zdiv - 1 ) posz = Zdiv - 1;

		
		int j1 = 0;
		int j2 = 0;
		int j3 = 0;
		
		
		//printf(" my min x is %f, and posx:	%d, posy:	%d, posz:	%d\n",Subdivision_min.x, posx,posy,posz);
		
		for (  int i = -1; i < 2 ; ++i ){
				
			j1 = posx + i;
			if (j1 > Xdiv - 1 || j1 < 0) continue; 

			for (  int j = -1; j < 2; ++j ){
					
				j2 = posy + j;
				if(j2 < 0 || j2 > Ydiv-1) continue;
				
	
				for (  int k = -1 ; k < 2; ++k ){
			
					j3 = posz + k;
					if(j3 < 0 || j3 > Zdiv-1) continue;
		

					int index = atomicAdd( &d_NoofNNlist[j3*Xdiv*Ydiv+j2*Xdiv+j1] , 1); //returns old
			  		
#ifdef PRINT_TOO_SHORT_ERROR
					if ( index > MaxNeighList )
					{
                				printf("Fullerene %d, NN-list too short, atleast %d\n", fullerene, index);
                      				 // for ( int k = 0; k < 32; ++k )
                      				 //     printf("%d ",d_NNlist[ 32*(j2*Xdiv+j1) + k]); 
                      				 // printf("\n");
						continue;
					}
#endif
					d_NNlist[ MaxNeighList*(j3*Xdiv*Ydiv+j2*Xdiv+j1)+index] = fullerene;
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
      		
      			
      		if ( C < Sub_min - 0.05 ) {
	
	  			int index = atomicAdd(d_counter_mc_l,1);
	 			d_migrated_cells_ind_L[index] = fullerene;
	 			d_cell_mig[fullerene] = 1;
	  		
	  	}
	  	
	  		
	  	if ( C > Sub_max + 0.05 ) {
	  		
	  			int index = atomicAdd(d_counter_mc_r,1);
	 			d_migrated_cells_ind_R[index] = fullerene;
	 			d_cell_mig[fullerene] = 1;
	  		
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
                               		float* d_Apo_rate_mc_buffer, float* d_squeeze_rate_mc_buffer, bool colloidal_dynamics)
{
	
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

				d_Youngs_mod_mc_buffer[cell]  = d_Youngs_mod[migrated_cell];
				d_Youngs_mod[migrated_cell]  = d_Youngs_mod[moving_Cell];
				
				d_ScaleFactor_mc_buffer[cell] = d_ScaleFactor[migrated_cell];
				d_ScaleFactor[migrated_cell] = d_ScaleFactor[moving_Cell];
				
				d_gamma_env_mc_buffer[cell] = d_gamma_env[migrated_cell];
				d_gamma_env[migrated_cell] = d_gamma_env[moving_Cell];
				
				d_viscotic_damp_mc_buffer[cell] = d_viscotic_damp[migrated_cell];
				d_viscotic_damp[migrated_cell] = d_viscotic_damp[moving_Cell];

				d_pressList_mc_buffer[cell] = d_pressList[migrated_cell];
				d_pressList[migrated_cell] = d_pressList[moving_Cell];
				
				d_CellINdex_mc_buffer[cell] = d_CellINdex[migrated_cell];
				d_CellINdex[migrated_cell] = d_CellINdex[moving_Cell];

				if(!colloidal_dynamics) {
								
					d_Growth_rate_mc_buffer[cell] = d_Growth_rate[migrated_cell];
					d_Growth_rate[migrated_cell] = d_Growth_rate[moving_Cell];
				
					d_DivisionVolume_mc_buffer[cell] = d_DivisionVolume[migrated_cell];
					d_DivisionVolume[migrated_cell] = d_DivisionVolume[moving_Cell];
								
					d_Apo_rate_mc_buffer[cell] = d_Apo_rate[migrated_cell];
					d_Apo_rate[migrated_cell] = d_Apo_rate[moving_Cell];
				
					d_squeeze_rate_mc_buffer[cell] = d_squeeze_rate[migrated_cell];
					d_squeeze_rate[migrated_cell] = d_squeeze_rate[moving_Cell];
		
		
				}		
		
		
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
				d_Youngs_mod_mc_buffer[cell]  = d_Youngs_mod[migrated_cell];
				d_ScaleFactor_mc_buffer[cell] = d_ScaleFactor[migrated_cell];
				d_gamma_env_mc_buffer[cell] = d_gamma_env[migrated_cell];
				d_viscotic_damp_mc_buffer[cell] = d_viscotic_damp[migrated_cell];
				d_pressList_mc_buffer[cell] = d_pressList[migrated_cell];
				d_CellINdex_mc_buffer[cell] = d_CellINdex[migrated_cell];
				
				if(!colloidal_dynamics) {
					
					d_Growth_rate_mc_buffer[cell] = d_Growth_rate[migrated_cell];
					d_DivisionVolume_mc_buffer[cell] = d_DivisionVolume[migrated_cell];
					d_Apo_rate_mc_buffer[cell] = d_Apo_rate[migrated_cell];
					d_squeeze_rate_mc_buffer[cell] = d_squeeze_rate[migrated_cell];
				}
		
			}			


	
		}
	
	
	}
		
}  


__global__ void migrated_Cells_Remove_Pack_PBC_X(int No_of_C180s, int No_of_migration_cells_buffer, int* d_counter,
					 	int* d_migrated_cells_ind, char* d_cell_mig, double3 boxMax,
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
                               		float* d_Apo_rate_mc_buffer, float* d_squeeze_rate_mc_buffer, bool colloidal_dynamics)
{

	
	int migrated_cell = d_migrated_cells_ind[blockIdx.x];
	
	int tid = threadIdx.x;	
	int cell = blockIdx.x;	
	
	if( cell < No_of_migration_cells_buffer ){

		__shared__ float C;
		
		if(tid == 0) {
		
			C = d_CMx[migrated_cell];
		
		}
		
		__syncthreads();
		
		
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
			
			
			
			d_X_mc_buffer[cell*192 + tid] = d_X[192*migrated_cell + tid] - floor( C/boxMax.x)*boxMax.x;
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
			
				
				d_CMx_mc_buffer[cell] = C - floor( C/boxMax.x)*boxMax.x;
				d_CMx[migrated_cell] = d_CMx[moving_Cell];
				
				d_CMy_mc_buffer[cell] = d_CMy[migrated_cell];
				d_CMy[migrated_cell] = d_CMy[moving_Cell];
				
				d_CMz_mc_buffer[cell] = d_CMz[migrated_cell];
				d_CMz[migrated_cell] = d_CMz[moving_Cell];

				d_Youngs_mod_mc_buffer[cell]  = d_Youngs_mod[migrated_cell];
				d_Youngs_mod[migrated_cell]  = d_Youngs_mod[moving_Cell];
				
				d_ScaleFactor_mc_buffer[cell] = d_ScaleFactor[migrated_cell];
				d_ScaleFactor[migrated_cell] = d_ScaleFactor[moving_Cell];
				
				d_gamma_env_mc_buffer[cell] = d_gamma_env[migrated_cell];
				d_gamma_env[migrated_cell] = d_gamma_env[moving_Cell];
				
				d_viscotic_damp_mc_buffer[cell] = d_viscotic_damp[migrated_cell];
				d_viscotic_damp[migrated_cell] = d_viscotic_damp[moving_Cell];

				d_pressList_mc_buffer[cell] = d_pressList[migrated_cell];
				d_pressList[migrated_cell] = d_pressList[moving_Cell];
				
				d_CellINdex_mc_buffer[cell] = d_CellINdex[migrated_cell];
				d_CellINdex[migrated_cell] = d_CellINdex[moving_Cell];

				if(!colloidal_dynamics) {
								
					d_Growth_rate_mc_buffer[cell] = d_Growth_rate[migrated_cell];
					d_Growth_rate[migrated_cell] = d_Growth_rate[moving_Cell];
				
					d_DivisionVolume_mc_buffer[cell] = d_DivisionVolume[migrated_cell];
					d_DivisionVolume[migrated_cell] = d_DivisionVolume[moving_Cell];
								
					d_Apo_rate_mc_buffer[cell] = d_Apo_rate[migrated_cell];
					d_Apo_rate[migrated_cell] = d_Apo_rate[moving_Cell];
				
					d_squeeze_rate_mc_buffer[cell] = d_squeeze_rate[migrated_cell];
					d_squeeze_rate[migrated_cell] = d_squeeze_rate[moving_Cell];
		
		
				}		
		
		
			}
		
		
		} else {

	
	
			d_X_mc_buffer[cell*192 + tid] = d_X[192*migrated_cell + tid] - floor( C/boxMax.x)*boxMax.x;
			d_Y_mc_buffer[cell*192 + tid] = d_Y[192*migrated_cell + tid];
			d_Z_mc_buffer[cell*192 + tid] = d_Z[192*migrated_cell + tid];
	
			d_velListX_mc_buffer[cell*192 + tid] = d_velListX[192*migrated_cell + tid];
			d_velListY_mc_buffer[cell*192 + tid] = d_velListY[192*migrated_cell + tid];
			d_velListZ_mc_buffer[cell*192 + tid] = d_velListZ[192*migrated_cell + tid]; 
		
		
			if(tid == 0){
			
				d_CMx_mc_buffer[cell] = C - floor( C / boxMax.x)*boxMax.x;
				d_CMy_mc_buffer[cell] = d_CMy[migrated_cell];
				d_CMz_mc_buffer[cell] = d_CMz[migrated_cell];
				d_Youngs_mod_mc_buffer[cell]  = d_Youngs_mod[migrated_cell];
				d_ScaleFactor_mc_buffer[cell] = d_ScaleFactor[migrated_cell];
				d_gamma_env_mc_buffer[cell] = d_gamma_env[migrated_cell];
				d_viscotic_damp_mc_buffer[cell] = d_viscotic_damp[migrated_cell];
				d_pressList_mc_buffer[cell] = d_pressList[migrated_cell];
				d_CellINdex_mc_buffer[cell] = d_CellINdex[migrated_cell];
				
				if(!colloidal_dynamics) {
					
					d_Growth_rate_mc_buffer[cell] = d_Growth_rate[migrated_cell];
					d_DivisionVolume_mc_buffer[cell] = d_DivisionVolume[migrated_cell];
					d_Apo_rate_mc_buffer[cell] = d_Apo_rate[migrated_cell];
					d_squeeze_rate_mc_buffer[cell] = d_squeeze_rate[migrated_cell];
				}
		
			}			


	
		}
	
	
	}
		
}  



__global__ void migrated_Cells_Remove_Pack_PBC_Y(int No_of_C180s, int No_of_migration_cells_buffer, int* d_counter,
					 	int* d_migrated_cells_ind, char* d_cell_mig, double3 boxMax,
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
                               		float* d_Apo_rate_mc_buffer, float* d_squeeze_rate_mc_buffer, bool colloidal_dynamics)
{

	
	
	int migrated_cell = d_migrated_cells_ind[blockIdx.x];
	
	int tid = threadIdx.x;	
	int cell = blockIdx.x;	
	
	if( cell < No_of_migration_cells_buffer ) {

		__shared__ float C;
		
		if(tid == 0) {
		
			C = d_CMy[migrated_cell]; 
		
		}
		
		__syncthreads();
		
		
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
			
			d_Y_mc_buffer[cell*192 + tid] = d_Y[192*migrated_cell + tid] - floor( C/boxMax.y)*boxMax.y;
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
				
				d_CMy_mc_buffer[cell] = C - floor( C/boxMax.y)*boxMax.y;
				d_CMy[migrated_cell] = d_CMy[moving_Cell];
				
				d_CMz_mc_buffer[cell] = d_CMz[migrated_cell];
				d_CMz[migrated_cell] = d_CMz[moving_Cell];

				d_Youngs_mod_mc_buffer[cell]  = d_Youngs_mod[migrated_cell];
				d_Youngs_mod[migrated_cell]  = d_Youngs_mod[moving_Cell];
				
				d_ScaleFactor_mc_buffer[cell] = d_ScaleFactor[migrated_cell];
				d_ScaleFactor[migrated_cell] = d_ScaleFactor[moving_Cell];
				
				d_gamma_env_mc_buffer[cell] = d_gamma_env[migrated_cell];
				d_gamma_env[migrated_cell] = d_gamma_env[moving_Cell];
				
				d_viscotic_damp_mc_buffer[cell] = d_viscotic_damp[migrated_cell];
				d_viscotic_damp[migrated_cell] = d_viscotic_damp[moving_Cell];

				d_pressList_mc_buffer[cell] = d_pressList[migrated_cell];
				d_pressList[migrated_cell] = d_pressList[moving_Cell];
				
				d_CellINdex_mc_buffer[cell] = d_CellINdex[migrated_cell];
				d_CellINdex[migrated_cell] = d_CellINdex[moving_Cell];

				if(!colloidal_dynamics) {
								
					d_Growth_rate_mc_buffer[cell] = d_Growth_rate[migrated_cell];
					d_Growth_rate[migrated_cell] = d_Growth_rate[moving_Cell];
				
					d_DivisionVolume_mc_buffer[cell] = d_DivisionVolume[migrated_cell];
					d_DivisionVolume[migrated_cell] = d_DivisionVolume[moving_Cell];
								
					d_Apo_rate_mc_buffer[cell] = d_Apo_rate[migrated_cell];
					d_Apo_rate[migrated_cell] = d_Apo_rate[moving_Cell];
				
					d_squeeze_rate_mc_buffer[cell] = d_squeeze_rate[migrated_cell];
					d_squeeze_rate[migrated_cell] = d_squeeze_rate[moving_Cell];
		
		
				}		
		
		
			}
		
		
		} else {
		
	
			d_X_mc_buffer[cell*192 + tid] = d_X[192*migrated_cell + tid];
			d_Y_mc_buffer[cell*192 + tid] = d_Y[192*migrated_cell + tid] - floor( C/boxMax.y)*boxMax.y;
			d_Z_mc_buffer[cell*192 + tid] = d_Z[192*migrated_cell + tid];
	
			d_velListX_mc_buffer[cell*192 + tid] = d_velListX[192*migrated_cell + tid];
			d_velListY_mc_buffer[cell*192 + tid] = d_velListY[192*migrated_cell + tid];
			d_velListZ_mc_buffer[cell*192 + tid] = d_velListZ[192*migrated_cell + tid]; 
		
		
			if(tid == 0){
			
				d_CMx_mc_buffer[cell] = d_CMx[migrated_cell];
				d_CMy_mc_buffer[cell] = C - floor( C / boxMax.y)*boxMax.y;
				d_CMz_mc_buffer[cell] = d_CMz[migrated_cell];
				d_Youngs_mod_mc_buffer[cell]  = d_Youngs_mod[migrated_cell];
				d_ScaleFactor_mc_buffer[cell] = d_ScaleFactor[migrated_cell];
				d_gamma_env_mc_buffer[cell] = d_gamma_env[migrated_cell];
				d_viscotic_damp_mc_buffer[cell] = d_viscotic_damp[migrated_cell];
				d_pressList_mc_buffer[cell] = d_pressList[migrated_cell];
				d_CellINdex_mc_buffer[cell] = d_CellINdex[migrated_cell];
				
				if(!colloidal_dynamics) {
					
					d_Growth_rate_mc_buffer[cell] = d_Growth_rate[migrated_cell];
					d_DivisionVolume_mc_buffer[cell] = d_DivisionVolume[migrated_cell];
					d_Apo_rate_mc_buffer[cell] = d_Apo_rate[migrated_cell];
					d_squeeze_rate_mc_buffer[cell] = d_squeeze_rate[migrated_cell];
				}
		
			}			


	
		}
	
	
	}
		
}  


__global__ void migrated_Cells_Remove_Pack_PBC_Z(int No_of_C180s, int No_of_migration_cells_buffer, int* d_counter,
					 	int* d_migrated_cells_ind, char* d_cell_mig, double3 boxMax,
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
                               		float* d_Apo_rate_mc_buffer, float* d_squeeze_rate_mc_buffer, bool colloidal_dynamics)
{

	
	
	int migrated_cell = d_migrated_cells_ind[blockIdx.x];
	
	int tid = threadIdx.x;	
	int cell = blockIdx.x;	
	
	if( cell < No_of_migration_cells_buffer ) {

		__shared__ float C;
		
		if(tid == 0) {

			C = d_CMz[migrated_cell]; 
		
		}
		
		__syncthreads();
		
		
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
			
			d_Z_mc_buffer[cell*192 + tid] = d_Z[192*migrated_cell + tid] - floor( C/boxMax.z)*boxMax.z;
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
				
				d_CMz_mc_buffer[cell] = C - floor( C/boxMax.z)*boxMax.z;
				d_CMz[migrated_cell] = d_CMz[moving_Cell];

				d_Youngs_mod_mc_buffer[cell]  = d_Youngs_mod[migrated_cell];
				d_Youngs_mod[migrated_cell]  = d_Youngs_mod[moving_Cell];
				
				d_ScaleFactor_mc_buffer[cell] = d_ScaleFactor[migrated_cell];
				d_ScaleFactor[migrated_cell] = d_ScaleFactor[moving_Cell];
				
				d_gamma_env_mc_buffer[cell] = d_gamma_env[migrated_cell];
				d_gamma_env[migrated_cell] = d_gamma_env[moving_Cell];
				
				d_viscotic_damp_mc_buffer[cell] = d_viscotic_damp[migrated_cell];
				d_viscotic_damp[migrated_cell] = d_viscotic_damp[moving_Cell];

				d_pressList_mc_buffer[cell] = d_pressList[migrated_cell];
				d_pressList[migrated_cell] = d_pressList[moving_Cell];
				
				d_CellINdex_mc_buffer[cell] = d_CellINdex[migrated_cell];
				d_CellINdex[migrated_cell] = d_CellINdex[moving_Cell];

				if(!colloidal_dynamics) {
								
					d_Growth_rate_mc_buffer[cell] = d_Growth_rate[migrated_cell];
					d_Growth_rate[migrated_cell] = d_Growth_rate[moving_Cell];
				
					d_DivisionVolume_mc_buffer[cell] = d_DivisionVolume[migrated_cell];
					d_DivisionVolume[migrated_cell] = d_DivisionVolume[moving_Cell];
								
					d_Apo_rate_mc_buffer[cell] = d_Apo_rate[migrated_cell];
					d_Apo_rate[migrated_cell] = d_Apo_rate[moving_Cell];
				
					d_squeeze_rate_mc_buffer[cell] = d_squeeze_rate[migrated_cell];
					d_squeeze_rate[migrated_cell] = d_squeeze_rate[moving_Cell];
		
		
				}		
		
		
			}
		
		
		} else {

	
			d_X_mc_buffer[cell*192 + tid] = d_X[192*migrated_cell + tid];
			d_Y_mc_buffer[cell*192 + tid] = d_Y[192*migrated_cell + tid];
			d_Z_mc_buffer[cell*192 + tid] = d_Z[192*migrated_cell + tid] - floor( C/boxMax.z)*boxMax.z;
	
			d_velListX_mc_buffer[cell*192 + tid] = d_velListX[192*migrated_cell + tid];
			d_velListY_mc_buffer[cell*192 + tid] = d_velListY[192*migrated_cell + tid];
			d_velListZ_mc_buffer[cell*192 + tid] = d_velListZ[192*migrated_cell + tid]; 
		
		
			if(tid == 0){
			
				d_CMx_mc_buffer[cell] = d_CMx[migrated_cell];
				d_CMy_mc_buffer[cell] = d_CMy[migrated_cell];
				d_CMz_mc_buffer[cell] = C - floor( C / boxMax.z)*boxMax.z;
				d_Youngs_mod_mc_buffer[cell]  = d_Youngs_mod[migrated_cell];
				d_ScaleFactor_mc_buffer[cell] = d_ScaleFactor[migrated_cell];
				d_gamma_env_mc_buffer[cell] = d_gamma_env[migrated_cell];
				d_viscotic_damp_mc_buffer[cell] = d_viscotic_damp[migrated_cell];
				d_pressList_mc_buffer[cell] = d_pressList[migrated_cell];
				d_CellINdex_mc_buffer[cell] = d_CellINdex[migrated_cell];
				
				if(!colloidal_dynamics) {
					
					d_Growth_rate_mc_buffer[cell] = d_Growth_rate[migrated_cell];
					d_DivisionVolume_mc_buffer[cell] = d_DivisionVolume[migrated_cell];
					d_Apo_rate_mc_buffer[cell] = d_Apo_rate[migrated_cell];
					d_squeeze_rate_mc_buffer[cell] = d_squeeze_rate[migrated_cell];
				}
		
			}			


	
		}
	
	
	}
		
}  

__global__ void migrated_Cells_Remove_Pack_LEbc_X(int No_of_C180s, int No_of_migration_cells_buffer, int* d_counter, float Pshift, float Vshift,
					 	int* d_migrated_cells_ind, char* d_cell_mig, double3 boxMax,
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
                               		float* d_Apo_rate_mc_buffer, float* d_squeeze_rate_mc_buffer, bool colloidal_dynamics)
{

	
	
	int migrated_cell = d_migrated_cells_ind[blockIdx.x];
	
	int tid = threadIdx.x;	
	int cell = blockIdx.x;	
	
	if( cell < No_of_migration_cells_buffer ) {

		__shared__ float Cx, Cy;
		
		if(tid == 0) {

			Cx = d_CMx[migrated_cell];
			Cy = d_CMy[migrated_cell]; 
		
		}
		
		__syncthreads();
		
		
		 int ModifierCx = floor( Cx/boxMax.x);
		 int ModifierCy = floor((Cy - ModifierCx*Pshift)/boxMax.y); 
		
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
			 
			
			d_X_mc_buffer[cell*192 + tid] = d_X[192*migrated_cell + tid] - ModifierCx*boxMax.x;
			d_X[migrated_cell*192 + tid] = d_X[192*moving_Cell + tid];
			
			d_Y_mc_buffer[cell*192 + tid] = d_Y[192*migrated_cell + tid] - ModifierCx*Pshift - ModifierCy*boxMax.y;
			d_Y[migrated_cell*192 + tid] = d_Y[192*moving_Cell + tid];
			
			d_Z_mc_buffer[cell*192 + tid] = d_Z[192*migrated_cell + tid];
			d_Z[migrated_cell*192 + tid] = d_Z[192*moving_Cell + tid];
			

			d_velListX_mc_buffer[cell*192 + tid] = d_velListX[192*migrated_cell + tid];
			d_velListX[migrated_cell*192 + tid] = d_velListX[192*moving_Cell + tid];
			
			d_velListY_mc_buffer[cell*192 + tid] = d_velListY[192*migrated_cell + tid] - ModifierCx*Vshift;
			d_velListY[migrated_cell*192 + tid] = d_velListY[192*moving_Cell + tid];
			
			
			d_velListZ_mc_buffer[cell*192 + tid] = d_velListZ[192*migrated_cell + tid]; 
			d_velListZ[migrated_cell*192 + tid] = d_velListZ[192*moving_Cell + tid];
			
			
			
			if(tid == 0){
			
				
				d_CMx_mc_buffer[cell] = Cx - ModifierCx*boxMax.x;
				d_CMx[migrated_cell] = d_CMx[moving_Cell];
				
				d_CMy_mc_buffer[cell] = Cy - ModifierCx*Pshift - ModifierCy*boxMax.y;
				d_CMy[migrated_cell] = d_CMy[moving_Cell];
				
				d_CMz_mc_buffer[cell] = d_CMz[migrated_cell];
				d_CMz[migrated_cell] = d_CMz[moving_Cell];

				d_Youngs_mod_mc_buffer[cell]  = d_Youngs_mod[migrated_cell];
				d_Youngs_mod[migrated_cell]  = d_Youngs_mod[moving_Cell];
				
				d_ScaleFactor_mc_buffer[cell] = d_ScaleFactor[migrated_cell];
				d_ScaleFactor[migrated_cell] = d_ScaleFactor[moving_Cell];
				
				d_gamma_env_mc_buffer[cell] = d_gamma_env[migrated_cell];
				d_gamma_env[migrated_cell] = d_gamma_env[moving_Cell];
				
				d_viscotic_damp_mc_buffer[cell] = d_viscotic_damp[migrated_cell];
				d_viscotic_damp[migrated_cell] = d_viscotic_damp[moving_Cell];

				d_pressList_mc_buffer[cell] = d_pressList[migrated_cell];
				d_pressList[migrated_cell] = d_pressList[moving_Cell];
				
				d_CellINdex_mc_buffer[cell] = d_CellINdex[migrated_cell];
				d_CellINdex[migrated_cell] = d_CellINdex[moving_Cell];

				if(!colloidal_dynamics) {
								
					d_Growth_rate_mc_buffer[cell] = d_Growth_rate[migrated_cell];
					d_Growth_rate[migrated_cell] = d_Growth_rate[moving_Cell];
				
					d_DivisionVolume_mc_buffer[cell] = d_DivisionVolume[migrated_cell];
					d_DivisionVolume[migrated_cell] = d_DivisionVolume[moving_Cell];
								
					d_Apo_rate_mc_buffer[cell] = d_Apo_rate[migrated_cell];
					d_Apo_rate[migrated_cell] = d_Apo_rate[moving_Cell];
				
					d_squeeze_rate_mc_buffer[cell] = d_squeeze_rate[migrated_cell];
					d_squeeze_rate[migrated_cell] = d_squeeze_rate[moving_Cell];
		
		
				}		
		
		
			}
		
		
		} else {

	
			d_X_mc_buffer[cell*192 + tid] = d_X[192*migrated_cell + tid] - ModifierCx*boxMax.x;
			d_Y_mc_buffer[cell*192 + tid] = d_Y[192*migrated_cell + tid] - ModifierCx*Pshift - ModifierCy*boxMax.y;
			d_Z_mc_buffer[cell*192 + tid] = d_Z[192*migrated_cell + tid];
	
			d_velListX_mc_buffer[cell*192 + tid] = d_velListX[192*migrated_cell + tid];
			d_velListY_mc_buffer[cell*192 + tid] = d_velListY[192*migrated_cell + tid] - ModifierCx*Vshift;
			d_velListZ_mc_buffer[cell*192 + tid] = d_velListZ[192*migrated_cell + tid]; 
		
		
			if(tid == 0){
			
				d_CMx_mc_buffer[cell] = Cx - ModifierCx*boxMax.x;
				d_CMy_mc_buffer[cell] = Cy - ModifierCx*Pshift - ModifierCy*boxMax.y;
				d_CMz_mc_buffer[cell] = d_CMz[migrated_cell];
				d_Youngs_mod_mc_buffer[cell]  = d_Youngs_mod[migrated_cell];
				d_ScaleFactor_mc_buffer[cell] = d_ScaleFactor[migrated_cell];
				d_gamma_env_mc_buffer[cell] = d_gamma_env[migrated_cell];
				d_viscotic_damp_mc_buffer[cell] = d_viscotic_damp[migrated_cell];
				d_pressList_mc_buffer[cell] = d_pressList[migrated_cell];
				d_CellINdex_mc_buffer[cell] = d_CellINdex[migrated_cell];
				
				if(!colloidal_dynamics) {
					
					d_Growth_rate_mc_buffer[cell] = d_Growth_rate[migrated_cell];
					d_DivisionVolume_mc_buffer[cell] = d_DivisionVolume[migrated_cell];
					d_Apo_rate_mc_buffer[cell] = d_Apo_rate[migrated_cell];
					d_squeeze_rate_mc_buffer[cell] = d_squeeze_rate[migrated_cell];
				}
		
			}			


	
		}
	
	
	}
		
}  


__global__ void ghost_cells_finder_Auxiliary(int No_of_C180s, int All_Cells, float *d_CM , 
						float Sub_max, float Sub_min, float R_ghost_buffer,
						int* d_counter_gc_r, int* d_counter_gc_l,
                         			int* d_Ghost_Cells_ind_R, int* d_Ghost_Cells_ind_L)
{
 
	int atom = blockIdx.x*blockDim.x+threadIdx.x;
	
	if ( atom < All_Cells )
	{	
		int fullerene = atom + No_of_C180s;
	 	
	 	float pos = d_CM[fullerene];
	 	
	 	if( pos <  Sub_min + R_ghost_buffer ){
	 			
	 		int index = atomicAdd(d_counter_gc_l,1);
	 		d_Ghost_Cells_ind_L[index] = fullerene;
	 	
	 	}
	 	
	 	if( pos >  Sub_max - R_ghost_buffer ){
	 			
	 		int index = atomicAdd(d_counter_gc_r,1);
	 		d_Ghost_Cells_ind_R[index] = fullerene;	 			
	 	}
	 	
	
	}
 
                         		 
}

__global__ void Ghost_Cells_Pack(int No_of_Ghost_cells_buffer, int* d_Ghost_Cells_ind,
				float *d_X,  float *d_Y,  float *d_Z,
                               float* d_velListX, float* d_velListY, float* d_velListZ,
                               float* d_CMx, float* d_CMy, float* d_CMz,
				float *d_X_gc_buffer,  float *d_Y_gc_buffer,  float *d_Z_gc_buffer,
                              float* d_velListX_gc_buffer, float* d_velListY_gc_buffer, float* d_velListZ_gc_buffer,
                              float* d_CMx_gc_buffer, float* d_CMy_gc_buffer, float* d_CMz_gc_buffer)
{

	
	
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

__global__ void Ghost_Cells_Pack_PBC_X(int No_of_Ghost_cells_buffer, int No_of_Ghost_cells_buffer_R, int* d_Ghost_Cells_ind, double3 boxMax,  float R_ghost_buffer,
					float *d_X,  float *d_Y,  float *d_Z,
                               	float* d_velListX, float* d_velListY, float* d_velListZ,
                               	float* d_CMx, float* d_CMy, float* d_CMz,
					float *d_X_gc_buffer,  float *d_Y_gc_buffer,  float *d_Z_gc_buffer,
                              	float* d_velListX_gc_buffer, float* d_velListY_gc_buffer, float* d_velListZ_gc_buffer,
                              	float* d_CMx_gc_buffer, float* d_CMy_gc_buffer, float* d_CMz_gc_buffer)
{


	int ghost_cell = d_Ghost_Cells_ind[blockIdx.x];
	
	int tid = threadIdx.x;	
	int cell = blockIdx.x;	
	
	if( cell < No_of_Ghost_cells_buffer ){
		
		__shared__ float Cx;
		
		
		if(tid == 0) Cx = d_CMx[ghost_cell]; 

		
		__syncthreads();
		
		
		int ModifierCx = floor((Cx - R_ghost_buffer - 0.2)/boxMax.x);
		if( cell < No_of_Ghost_cells_buffer_R ) ModifierCx = floor( (Cx + R_ghost_buffer + 0.2)/boxMax.x);
			
			
		d_X_gc_buffer[cell*192 + tid] = d_X[192*ghost_cell + tid] - ModifierCx*boxMax.x;
		d_Y_gc_buffer[cell*192 + tid] = d_Y[192*ghost_cell + tid];
		d_Z_gc_buffer[cell*192 + tid] = d_Z[192*ghost_cell + tid];
	
	
		d_velListX_gc_buffer[cell*192 + tid] = d_velListX[192*ghost_cell + tid];
		d_velListY_gc_buffer[cell*192 + tid] = d_velListY[192*ghost_cell + tid];
		d_velListZ_gc_buffer[cell*192 + tid] = d_velListZ[192*ghost_cell + tid]; 
		
		
		if(tid == 0){
			
			d_CMx_gc_buffer[cell] = Cx - ModifierCx*boxMax.x;
			d_CMy_gc_buffer[cell] = d_CMy[ghost_cell];
			d_CMz_gc_buffer[cell] = d_CMz[ghost_cell];
		
		
		}
		
	
	}		
	
} 

__global__ void Ghost_Cells_Pack_PBC_Y(int No_of_Ghost_cells_buffer, int No_of_Ghost_cells_buffer_R, int* d_Ghost_Cells_ind, double3 boxMax,  float R_ghost_buffer,
					float *d_X,  float *d_Y,  float *d_Z,
                               	float* d_velListX, float* d_velListY, float* d_velListZ,
                               	float* d_CMx, float* d_CMy, float* d_CMz,
					float *d_X_gc_buffer,  float *d_Y_gc_buffer,  float *d_Z_gc_buffer,
                              	float* d_velListX_gc_buffer, float* d_velListY_gc_buffer, float* d_velListZ_gc_buffer,
                              	float* d_CMx_gc_buffer, float* d_CMy_gc_buffer, float* d_CMz_gc_buffer)
{
	

	int ghost_cell = d_Ghost_Cells_ind[blockIdx.x];
	
	int tid = threadIdx.x;	
	int cell = blockIdx.x;	
	
	if( cell < No_of_Ghost_cells_buffer ){
		
		__shared__ float Cy;
		
		
		if(tid == 0) Cy = d_CMy[ghost_cell]; 

		
		__syncthreads();
		
		
		int ModifierCy = floor( (Cy - R_ghost_buffer - 0.2)/boxMax.y);
		if( cell < No_of_Ghost_cells_buffer_R ) ModifierCy = floor((Cy + R_ghost_buffer + 0.2)/boxMax.y);
			
			
		d_X_gc_buffer[cell*192 + tid] = d_X[192*ghost_cell + tid];
		d_Y_gc_buffer[cell*192 + tid] = d_Y[192*ghost_cell + tid] - ModifierCy*boxMax.y;
		d_Z_gc_buffer[cell*192 + tid] = d_Z[192*ghost_cell + tid];
	
	
		d_velListX_gc_buffer[cell*192 + tid] = d_velListX[192*ghost_cell + tid];
		d_velListY_gc_buffer[cell*192 + tid] = d_velListY[192*ghost_cell + tid];
		d_velListZ_gc_buffer[cell*192 + tid] = d_velListZ[192*ghost_cell + tid]; 
		
		
		if(tid == 0){
			
			d_CMx_gc_buffer[cell] = d_CMx[ghost_cell];
			d_CMy_gc_buffer[cell] = Cy - ModifierCy*boxMax.y;
			d_CMz_gc_buffer[cell] = d_CMz[ghost_cell];
		
		
		}
		
	
	}

} 


__global__ void Ghost_Cells_Pack_PBC_Z(int No_of_Ghost_cells_buffer, int No_of_Ghost_cells_buffer_R, int* d_Ghost_Cells_ind, double3 boxMax, float R_ghost_buffer,
					float *d_X,  float *d_Y,  float *d_Z,
                               	float* d_velListX, float* d_velListY, float* d_velListZ,
                               	float* d_CMx, float* d_CMy, float* d_CMz,
					float *d_X_gc_buffer,  float *d_Y_gc_buffer,  float *d_Z_gc_buffer,
                              	float* d_velListX_gc_buffer, float* d_velListY_gc_buffer, float* d_velListZ_gc_buffer,
                              	float* d_CMx_gc_buffer, float* d_CMy_gc_buffer, float* d_CMz_gc_buffer)
{


	int ghost_cell = d_Ghost_Cells_ind[blockIdx.x];
	
	int tid = threadIdx.x;	
	int cell = blockIdx.x;	
	
	if( cell < No_of_Ghost_cells_buffer ){
		
		__shared__ float Cz;
		
		if(tid == 0) Cz = d_CMz[ghost_cell]; 

		__syncthreads();
		
		int ModifierCz = floor( (Cz - R_ghost_buffer - 0.2)/boxMax.z);
		if( cell < No_of_Ghost_cells_buffer_R ) ModifierCz = floor((Cz + R_ghost_buffer + 0.2)/boxMax.z);
			
			
		d_X_gc_buffer[cell*192 + tid] = d_X[192*ghost_cell + tid];
		d_Y_gc_buffer[cell*192 + tid] = d_Y[192*ghost_cell + tid];
		d_Z_gc_buffer[cell*192 + tid] = d_Z[192*ghost_cell + tid] - ModifierCz*boxMax.z;
	
	
		d_velListX_gc_buffer[cell*192 + tid] = d_velListX[192*ghost_cell + tid];
		d_velListY_gc_buffer[cell*192 + tid] = d_velListY[192*ghost_cell + tid];
		d_velListZ_gc_buffer[cell*192 + tid] = d_velListZ[192*ghost_cell + tid]; 
		
		
		if(tid == 0){
			
			d_CMx_gc_buffer[cell] = d_CMx[ghost_cell];
			d_CMy_gc_buffer[cell] = d_CMy[ghost_cell];
			d_CMz_gc_buffer[cell] = Cz - ModifierCz*boxMax.z;
		
		
		}		
	
	}	

}


__global__ void Ghost_Cells_Pack_LEbc_X(int No_of_Ghost_cells_buffer, int No_of_Ghost_cells_buffer_R, int* d_Ghost_Cells_ind, double3 boxMax, float R_ghost_buffer,
					float Pshift, float Vshift,
					float *d_X,  float *d_Y,  float *d_Z,
                               	float* d_velListX, float* d_velListY, float* d_velListZ,
                               	float* d_CMx, float* d_CMy, float* d_CMz,
					float *d_X_gc_buffer,  float *d_Y_gc_buffer,  float *d_Z_gc_buffer,
                              	float* d_velListX_gc_buffer, float* d_velListY_gc_buffer, float* d_velListZ_gc_buffer,
                              	float* d_CMx_gc_buffer, float* d_CMy_gc_buffer, float* d_CMz_gc_buffer)
{

	
	int ghost_cell = d_Ghost_Cells_ind[blockIdx.x];
	
	int tid = threadIdx.x;	
	int cell = blockIdx.x;	
	
	if( cell < No_of_Ghost_cells_buffer ) {
		
		__shared__ float Cx, Cy;
		
		
		if(tid == 0) {
			
			Cx = d_CMx[ghost_cell]; 
			Cy = d_CMy[ghost_cell];
		}
		
		__syncthreads();
		
		
		int ModifierCx = floor( (Cx - R_ghost_buffer - 0.2)/boxMax.x);
		if( cell < No_of_Ghost_cells_buffer_R ) ModifierCx = floor( (Cx + R_ghost_buffer + 0.2)/boxMax.x);
		
		
		int ModifierCy = floor((Cy - ModifierCx*Pshift)/boxMax.y);
			
			
		d_X_gc_buffer[cell*192 + tid] = d_X[192*ghost_cell + tid] - ModifierCx*boxMax.x;
		d_Y_gc_buffer[cell*192 + tid] = d_Y[192*ghost_cell + tid] - ModifierCx*Pshift - ModifierCy*boxMax.y;
		d_Z_gc_buffer[cell*192 + tid] = d_Z[192*ghost_cell + tid];
	
	
		d_velListX_gc_buffer[cell*192 + tid] = d_velListX[192*ghost_cell + tid];
		d_velListY_gc_buffer[cell*192 + tid] = d_velListY[192*ghost_cell + tid] - ModifierCx*Vshift;
		d_velListZ_gc_buffer[cell*192 + tid] = d_velListZ[192*ghost_cell + tid]; 
		
		
		if(tid == 0){
			
			d_CMx_gc_buffer[cell] = Cx - ModifierCx*boxMax.x;
			d_CMy_gc_buffer[cell] = Cy - ModifierCx*Pshift - ModifierCy*boxMax.y;
			d_CMz_gc_buffer[cell] = d_CMz[ghost_cell];
		
		
		}
		
	
	}	
	
}

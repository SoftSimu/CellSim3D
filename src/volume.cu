#include <cuda.h>
#include <stdio.h>
#include "VectorFunctions.hpp"
#include<curand_kernel.h>

__global__ void volumes( int No_of_C180s, int *C180_56,
                         float *X,    float *Y,   float *Z,
                         float *CMx , float *CMy, float *CMz, float *vol,
                         char* cell_div, float* d_DivisionVolume, bool checkSphericity,
                         float* areaList,
                         float stiffness1, bool useDifferentCell, float* d_younds_mod, float* d_Growth_rate,
                         bool recalc_r0, float ApoVol, float* d_ScaleFactor,
                         int *num_cell_div, int *cell_div_inds, char* d_cell_Apo, int* d_num_cell_Apo, int *d_cell_Apo_inds){
                         
    __shared__ float locX[192];
    __shared__ float locY[192];
    __shared__ float locZ[192];
    __shared__ float volume;
    __shared__ float area; 

    int fullerene = blockIdx.x;
    int tid       = threadIdx.x;

    if ( tid < 180 ){
    
        locX[tid] = X[192*fullerene+tid] -CMx[fullerene];
        locY[tid] = Y[192*fullerene+tid] -CMy[fullerene];
        locZ[tid] = Z[192*fullerene+tid] -CMz[fullerene];
    }


    if ( tid == 0){
        volume = 0.0f;
        area = 0.0f;
    }

    __syncthreads();

    if ( tid < 92 ){

        float avX = 0.0f;
        float avY = 0.0f;
        float avZ = 0.0f;

        for ( int i = 0; i < 5; ++i ){
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
        float faceArea = 0.0f;


        for ( int i = 0; i < 6; ++i ){
            n1 = (locY[C180_56[7*tid+i+1]]*avZ-avY*locZ[C180_56[7*tid+i+1]])*locX[C180_56[7*tid+i]];
            n2 = (locZ[C180_56[7*tid+i+1]]*avX-avZ*locX[C180_56[7*tid+i+1]])*locY[C180_56[7*tid+i]];
            n3 = (locX[C180_56[7*tid+i+1]]*avY-avX*locY[C180_56[7*tid+i+1]])*locZ[C180_56[7*tid+i]];
            totvol += fabsf(n1+n2+n3);



            if (checkSphericity){
       
                // Get vectors that define a triangle 1, 2
                float x1 = locX[C180_56[7*tid+i]] - avX;
                float y1 = locY[C180_56[7*tid+i]] - avY;
                float z1 = locZ[C180_56[7*tid+i]] - avZ;

                float x2 = locX[C180_56[7*tid+i+1]] - avX;
                float y2 = locY[C180_56[7*tid+i+1]] - avY;
                float z2 = locZ[C180_56[7*tid+i+1]] - avZ;

                // now 1 will hold 1X2
                float xx = y1*z2 - z1*y2;
                float yy = z1*x2 - x1*z2;
                float zz = x1*y2 - y1*x2;

                // area of triangle is then 0.5*|1|
                faceArea += 0.5 * sqrtf(xx*xx + yy*yy + zz*zz);
            }
        }
        atomicAdd(&volume, totvol);
    
        if (checkSphericity)
            atomicAdd(&area, faceArea); 
    }

    __syncthreads();

    if ( tid == 0){
        volume = volume/6.0;
        vol[fullerene] = volume;
        
        if (!isfinite(volume)){
            printf("OH SHIT: non-finite volume %f, cell %d\n", volume, fullerene);
            printf("Crash now :(\n");
            asm("trap;");
            volume = 1.f;
        }
        
        if (volume >  d_DivisionVolume[fullerene]){
            cell_div[fullerene] = 1;
        }
	

	if ( volume < ApoVol ){
	    	
	    	if (d_Growth_rate[fullerene] < 0){
	    		
	    		int index = atomicAdd(&d_num_cell_Apo[0],1);
	    		d_cell_Apo_inds[index] = fullerene;
	    		d_cell_Apo[fullerene] = 1;
	    	}	
	} 


        if (checkSphericity){
            areaList[fullerene] = area;
            float c = cbrtf(volume);
            float psi = 4.835975862049408 * c * c/area;
            if (abs(1.0f - psi) > 0.05){ // why 0.05?
                cell_div[fullerene] = 0;
                //printf("cell %d division rejected\n", fullerene);
            }
        }

        if (useDifferentCell){
            if (recalc_r0){
                if (d_younds_mod[fullerene] != stiffness1){
                    cell_div[fullerene] = 0;
                }
            }
        }
    
        if ( cell_div[fullerene] == 1 )                      
	{                                                    
		int index = atomicAdd(&num_cell_div[0],1);   
		cell_div_inds[index] = fullerene;            
	} 
    
    
    }
}

__device__ void CalcAndUpdateDaughtPos(int daughtInd, int partInd, float halfGap,
                                       float CMx, float CMy, float CMz,
                                       float X, float Y, float Z,
                                       float* d_X,  float* d_Y,  float* d_Z,
                                       float planeNx, float planeNy, float planeNz){
             
    // redefine position of parent cell wrt to an origin that includes
    // 0.5 the minimum gap between daughter cells
    X = X - CMx - halfGap*planeNx; 
    Y = Y - CMy - halfGap*planeNy; 
    Z = Z - CMz - halfGap*planeNz;

    float posDotN = X*planeNx + Y*planeNy + Z*planeNz;

    // If particle is below the plane, project onto the plane
    if (posDotN < 0.0f || posDotN == 0){
        X = X - posDotN*planeNx;
        Y = Y - posDotN*planeNy;
        Z = Z - posDotN*planeNz;
    }
    
    d_X[daughtInd*192+partInd] = X + (CMx + halfGap*planeNx);
    d_Y[daughtInd*192+partInd] = Y + (CMy + halfGap*planeNy);
    d_Z[daughtInd*192+partInd] = Z + (CMz + halfGap*planeNz);
    
}


__global__ void  cell_division( 
                               float *d_X,  float *d_Y,  float *d_Z,
                               float* AllCMx, float* AllCMy, float* AllCMz,
                               float* d_velListX, float* d_velListY, float* d_velListZ, 
                               int No_of_C180s, float repulsion_range, float* d_asym,
                               bool useDifferentCell, bool daughtSame,
                               int NewCellInd, float stiffness1, float rMax, float divVol, float gamma_visc, float viscotic_damping,
                               float squeeze_rate1, float Apo_rate1,
                               float* d_ScaleFactor,float* d_Youngs_mod, float* d_Growth_rate, float* d_DivisionVolume,
                               float* d_squeeze_rate, float* d_Apo_rate,
                               float* d_gamma_env, float* d_viscotic_damp, int* d_CellINdex,
				R3Nptrs d_DivPlane, int *num_cell_div, int *cell_div_inds, float *pressList, float minPressure){ 
   
         
    __shared__ float CMx, CMy, CMz;
    
    int tid  = threadIdx.x;
    int atom = tid;
    int bid  = blockIdx.x;
    
    int rank    = cell_div_inds[bid]; 
    int newrank = No_of_C180s+bid;    

    if (tid == 0){
    
        //index = atomicAdd( d_division_counter, 1);
        //int i = atomicAdd( d_No_of_C180s, 1);
        //printf("Num of Cells:		%d\n",i);
        CMx = AllCMx[rank];
        CMy = AllCMy[rank];
        CMz = AllCMz[rank];
        
    }

    __syncthreads();


    if ( atom < 180 ) 
    {

        // planeN is the division plane's normal vector
        //loat planeNx = d_randNorm[0];
        //float planeNy = d_randNorm[1];
        //float planeNz = d_randNorm[2];
        
        
        float planeNx = d_DivPlane.x[newrank];                 
        float planeNy = d_DivPlane.y[newrank];                 
        float planeNz = d_DivPlane.z[newrank];                 
        float asym = d_asym[newrank];                          


        if (abs(sqrt(planeNx*planeNx + planeNy*planeNy + planeNz*planeNz) - 1) > 1e-3){
            printf("OH SHIT: normal is not normalized\n");
            printf("Crash now :(\n"); 
            asm("trap;");
        }


        // First generate and write positions for the first daughter

        float X = d_X[rank*192+atom]; 
        float Y = d_Y[rank*192+atom]; 
        float Z = d_Z[rank*192+atom]; 
        
        
        //if (tid == 0) printf("half gap:	%f\n",(1-asym)*repulsion_range);
        

        
        CalcAndUpdateDaughtPos(rank, atom, (1-asym)*repulsion_range,
                               CMx, CMy, CMz,
                               X, Y, Z, 
                               d_X, d_Y, d_Z, 
                               planeNx, planeNy, planeNz);

        
        // Invert the normal
        planeNx = -1*planeNx; 
        planeNy = -1*planeNy; 
        planeNz = -1*planeNz;

        // Now repeat for the second daughter
        CalcAndUpdateDaughtPos(newrank, atom, asym*repulsion_range,
                               CMx, CMy, CMz,
                               X, Y, Z, 
                               d_X, d_Y, d_Z,
                               planeNx, planeNy, planeNz);

        // give the daughter the same velocities as the parent
        d_velListX[newrank*192 + atom] = d_velListX[rank*192+atom];
        d_velListY[newrank*192 + atom] = d_velListY[rank*192+atom];
        d_velListZ[newrank*192 + atom] = d_velListZ[rank*192+atom];
    
    }
    
    
    if (tid == 0){
    
    	if (useDifferentCell && daughtSame){
	
		d_ScaleFactor[newrank] = d_ScaleFactor[rank];
        	d_Youngs_mod[newrank] = d_Youngs_mod[rank];
        	d_Growth_rate[newrank] = d_Growth_rate[rank];
        	d_DivisionVolume[newrank] = d_DivisionVolume[rank];      
       	d_gamma_env[newrank] = d_gamma_env[rank];
       	d_viscotic_damp[newrank] = d_viscotic_damp[rank];
       	d_squeeze_rate[newrank] = d_squeeze_rate[rank];
          	d_Apo_rate[newrank] = d_Apo_rate[rank];
       	
       	if( d_CellINdex[rank] < 0 ){
       		d_CellINdex[newrank] = - (NewCellInd + bid);
       	} else {
       		d_CellINdex[newrank] = NewCellInd + bid;
       	}
        
        } else {
        
              	d_Youngs_mod[newrank] = stiffness1; 
          	d_ScaleFactor[newrank] = 1;
          	d_Growth_rate[newrank] = rMax;
          	d_DivisionVolume[newrank] = divVol;
          	d_gamma_env[newrank] = gamma_visc;
          	d_viscotic_damp[newrank] = viscotic_damping;
          	d_CellINdex[newrank] = NewCellInd + bid;
          	d_squeeze_rate[newrank] = squeeze_rate1;
          	d_Apo_rate[newrank] = Apo_rate1;
          	
          	
          	  	
        }
        
        pressList[rank] = minPressure;             
	pressList[newrank] = minPressure;          
    
    }
    
    
}
   
   
__global__ void Cell_removing (int No_of_C180s, int num_cell_Apo, int* d_counter,
				float *d_X,  float *d_Y,  float *d_Z,
                               float* d_velListX, float* d_velListY, float* d_velListZ, 
                               float* d_ScaleFactor,float* d_Youngs_mod, float* d_Growth_rate, float* d_DivisionVolume,
                               float* d_gamma_env, float* d_viscotic_damp, float* d_pressList, int* d_CellINdex, 
                               float* d_Apo_rate, float* d_squeeze_rate,
				int* d_cell_Apo_inds, char* cell_Apo){

	
	
	int dead_cell = d_cell_Apo_inds[blockIdx.x];
		
	if( dead_cell < No_of_C180s - num_cell_Apo ) {
	 
	 	__shared__ int moving_Cell;
		
		int tid = threadIdx.x;
		
		if (tid == 0){

			int index = atomicAdd(d_counter,1);
			moving_Cell = No_of_C180s - index - 1;			
		
			while ( cell_Apo[moving_Cell] == 1 ){
				
				index = atomicAdd(d_counter,1);
				moving_Cell = No_of_C180s - index - 1;
			}
	
		}
	
		__syncthreads();
	
	
		d_X[dead_cell*192 + tid] = d_X[192*moving_Cell + tid];
		d_Y[dead_cell*192 + tid] = d_Y[192*moving_Cell + tid];
		d_Z[dead_cell*192 + tid] = d_Z[192*moving_Cell + tid];
	
	
		d_velListX[dead_cell*192 + tid] = d_velListX[192*moving_Cell + tid];
		d_velListY[dead_cell*192 + tid] = d_velListY[192*moving_Cell + tid];
		d_velListZ[dead_cell*192 + tid] = d_velListZ[192*moving_Cell + tid]; 
	
	
		if(tid == 0){
		
			d_pressList[dead_cell] = d_pressList[moving_Cell];
			d_Growth_rate[dead_cell] = d_Growth_rate[moving_Cell];
			d_Youngs_mod[dead_cell]  = d_Youngs_mod[moving_Cell];
			d_ScaleFactor[dead_cell] = d_ScaleFactor[moving_Cell];
			d_DivisionVolume[dead_cell] = d_DivisionVolume[moving_Cell];
			d_gamma_env[dead_cell] = d_gamma_env[moving_Cell];
			d_viscotic_damp[dead_cell] = d_viscotic_damp[moving_Cell];
			d_CellINdex[dead_cell] = d_CellINdex[moving_Cell];
			d_Apo_rate[dead_cell] = d_Apo_rate[moving_Cell];
			d_squeeze_rate[dead_cell] = d_squeeze_rate[moving_Cell];
	
		}
	
	}
	
	
}  

__global__ void CellApoptosis(int No_of_C180s, curandState *d_rngStatesApo, float* d_Apo_rate,
 				float* d_Growth_rate, float* d_squeeze_rate, int* d_Num_shrink_Cell){
                                  
        size_t cell = blockIdx.x*blockDim.x + threadIdx.x;
    
    
    	if ( cell < No_of_C180s){

        	
      			       			
      		if (d_Growth_rate[cell] > 0) {
        	        
        	        curandState rngState = d_rngStatesApo[cell];
        	        float rand = curand_uniform(&rngState);
        	        
        	        if ( rand < d_Apo_rate[cell] ){
        	        
        	        	d_Growth_rate[cell] = d_squeeze_rate[cell];
        	        	atomicAdd(&d_Num_shrink_Cell[0],1);
        	        	
        	        	
        	        }
      			
      			d_rngStatesApo[cell] = rngState;
      		}
		
	}

}
  

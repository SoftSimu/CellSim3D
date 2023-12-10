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
                         int *d_num_cell_div, int *cell_div_inds, char* d_cell_Apo, int* d_num_cell_Apo, int *d_cell_Apo_inds){
                         
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

	    	

            //if (checkSphericity){
       
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
                
            //}
        
       }
        
        
        atomicAdd(&volume, totvol);
        atomicAdd(&area, faceArea);
        
        //if (checkSphericity) atomicAdd(&area, faceArea);     
    
    }

    __syncthreads();

    if ( tid == 0){
    
        volume = volume/6.0;
        vol[fullerene] = volume;
        areaList[fullerene] = area;
        
        
        if (!isfinite(volume)){
            printf("OH SHIT: non-finite volume %f, cell %d\n", volume, fullerene);
            printf("Crash now :(\n");
            asm("trap;");
            volume = 1.f;
        }
        
        if ( volume >  d_DivisionVolume[fullerene]){
            
            cell_div[fullerene] = 1;
            
            
            if (checkSphericity){
        
            	//areaList[fullerene] = area;
            	float c = cbrtf(volume);
            	float psi = 4.835975862049408 * c * c/area;
            	if (abs(1.0f - psi) > 0.05) cell_div[fullerene] = 0;

            }
            
            if (useDifferentCell && recalc_r0){
            	
            	    if (d_younds_mod[fullerene] != stiffness1) cell_div[fullerene] = 0;
            }
        
            
            if ( cell_div[fullerene] == 1 )                      
	     {                                                    
		
		int index = atomicAdd(&d_num_cell_div[0],1);   
		cell_div_inds[index] = fullerene;  
		cell_div[fullerene] = 0;          
		
	     } 
    	
        
        }
	

	if ( volume < ApoVol ){
	    	
	    	if (d_Growth_rate[fullerene] < 0){
	    		
	    		int index = atomicAdd(&d_num_cell_Apo[0],1);
	    		d_cell_Apo_inds[index] = fullerene;
	    		d_cell_Apo[fullerene] = 1;
	    	}	
	
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


__global__ void  cell_division( bool Random_Div_Rule, bool Fibre,
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
				R3Nptrs d_DivPlane, int *num_cell_div, int *cell_div_inds, float *pressList, int* d_Generation, int* d_Fibre_index,
				float minPressure){ 
   
         
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

        float planeNx, planeNy, planeNz;
        
        if(Random_Div_Rule){
        	
        	if(Fibre){

	        	int loc = d_Fibre_index[rank];
	        	planeNx = d_DivPlane.x[loc];                 
	        	planeNy = d_DivPlane.y[loc];                 
	        	planeNz = d_DivPlane.z[loc];                 
        	
        	
        	} else {
        	
        		planeNx = d_DivPlane.x[newrank];                 
        		planeNy = d_DivPlane.y[newrank];                 
        		planeNz = d_DivPlane.z[newrank];                 
        	
        	}
        
        } else {
        	
        	int loc = d_Generation[rank];
        	planeNx = d_DivPlane.x[loc];                 
        	planeNy = d_DivPlane.y[loc];                 
        	planeNz = d_DivPlane.z[loc];                 
        
        }
        
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
        
        d_Generation[rank] = d_Generation[rank] + 1;
        d_Generation[newrank] = d_Generation[rank];
        
        d_Fibre_index[newrank] = d_Fibre_index[rank];
         
    
    }
    
    
}
   
   
__global__ void Cell_removing (int No_of_C180s, int num_cell_Apo, int* d_counter,
				float *d_X,  float *d_Y,  float *d_Z,
                               float* d_velListX, float* d_velListY, float* d_velListZ, 
                               float* d_ScaleFactor,float* d_Youngs_mod, float* d_Growth_rate, float* d_DivisionVolume,
                               float* d_gamma_env, float* d_viscotic_damp, float* d_pressList, int* d_CellINdex, 
                               float* d_Apo_rate, float* d_squeeze_rate, int* d_Generation, int* d_Fibre_index,
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
			d_Generation[dead_cell] = d_Generation[moving_Cell];
			d_Fibre_index[dead_cell] = d_Fibre_index[moving_Cell]; 
	
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
  
  
__global__ void CellStressTensor( float *d_X,  float *d_Y,  float *d_Z,
				   float *d_CMx, float *d_CMy, float *d_CMz,
				   float *d_volume, R3Nptrs d_ExtForces,
				   float* d_Stress)
				   
{  


	__shared__ float  Sxx[256];
	__shared__ float  Sxy[256];
	__shared__ float  Sxz[256];
	
	__shared__ float  Syx[256];
	__shared__ float  Syy[256];
	__shared__ float  Syz[256];
	
	__shared__ float  Szx[256];
	__shared__ float  Szy[256];
	__shared__ float  Szz[256];


    	int rank = blockIdx.x;
    	int atom = threadIdx.x;
	long int atomInd = rank*192+atom;
	
	
	Sxx[atom] = 0.0;
	Sxy[atom] = 0.0;
	Sxz[atom] = 0.0;
	
	Syx[atom] = 0.0;
	Syy[atom] = 0.0;
	Syz[atom] = 0.0;
	
	Szx[atom] = 0.0;
	Szy[atom] = 0.0;
	Szz[atom] = 0.0;
	
	if (atom < 180){
	
		
		
		float3 r_CM = make_float3(d_X[atomInd] - d_CMx[rank], 
					   d_Y[atomInd] - d_CMy[rank],
					   d_Z[atomInd] - d_CMz[rank]);
					   
		float3 Force = make_float3(d_ExtForces.x[atomInd], 
					    d_ExtForces.y[atomInd],
					    d_ExtForces.z[atomInd]);
		
		Sxx[atom] = Force.x*r_CM.x;
		Sxy[atom] = Force.x*r_CM.y;
		Sxz[atom] = Force.x*r_CM.z;					   

		Syx[atom] = Force.y*r_CM.x;
		Syy[atom] = Force.y*r_CM.y;
		Syz[atom] = Force.y*r_CM.z;

		Szx[atom] = Force.z*r_CM.x;
		Szy[atom] = Force.z*r_CM.y;
		Szz[atom] = Force.z*r_CM.z;
		
		//printf("F_Z: %f, X : %f, Y : %f, Z : %f\n",Force.z,r_CM.x,r_CM.y,r_CM.z);

	}
				   
	__syncthreads();

	//if(atom == 0) printf("\n");

	for ( int s = blockDim.x/2; s > 0; s>>=1)
   	{
   		if ( atom < s )
      		{
      		
      			Sxx[atom] += Sxx[atom+s];
      			Sxy[atom] += Sxy[atom+s];
      			Sxz[atom] += Sxz[atom+s];
      			
      			Syx[atom] += Syx[atom+s];
      			Syy[atom] += Syy[atom+s];
      			Syz[atom] += Syz[atom+s];
      		
      			Szx[atom] += Szx[atom+s];
      			Szy[atom] += Szy[atom+s];
      			Szz[atom] += Szz[atom+s];
      		
      		
      		}
   		
   		__syncthreads();
   	
   	}

	
	if ( atom == 0 ) 
   	{
   		
   		float Vol = d_volume[rank];
   		
   		int shift = rank*32;
   			
   		d_Stress[shift + 0] = Sxx[0]/Vol;
   		d_Stress[shift + 1] = Sxy[0]/Vol;
   		d_Stress[shift + 2] = Sxz[0]/Vol;
   		
   		d_Stress[shift + 3] = Syx[0]/Vol;
   		d_Stress[shift + 4] = Syy[0]/Vol;
   		d_Stress[shift + 5] = Syz[0]/Vol;
   		
   		d_Stress[shift + 6] = Szx[0]/Vol;
   		d_Stress[shift + 7] = Szy[0]/Vol;
   		d_Stress[shift + 8] = Szz[0]/Vol;
   		
   		//printf("Matrix\n");
		//printf("%.4f, %.4f, %.4f\n", d_Stress[shift + 0], d_Stress[shift + 1], d_Stress[shift + 2]);
		//printf("%.4f, %.4f, %.4f\n", d_Stress[shift + 3], d_Stress[shift + 4], d_Stress[shift + 5]);
		//printf("%.4f, %.4f, %.4f\n", d_Stress[shift + 6], d_Stress[shift + 7], d_Stress[shift + 8]);
		//printf("\n");
   		
	}


}  


__global__ void PowerItr( int No_of_C180s, int step, float *d_Stress, R3Nptrs d_Polarity_Vec)
{				   

	
    	int tid = threadIdx.x;
    	int rank = blockIdx.x;
    	int tInd = blockIdx.x*blockDim.x + threadIdx.x;
    	
    	if (rank < No_of_C180s){ 


		__shared__ float matrix[9];
		__shared__ float eigenVector[3];
		__shared__ float prevEigenVector[3];
    		
    		float S = d_Stress[tInd];
    		float lambda = 0.0;
    	 
		if (tid < 9){
	
			matrix[tid] = S;
			if (tid < 3) eigenVector[tid] = - 1.0;
	
		}
	
		__syncthreads();
	
	
		if(tid == 0 ){
			

			
			
		}
	
	
		if (tid < 3){
	
		
			float normDiff = 1.0;
			float norm = 1.0;
		
			while (normDiff > 1e-2)
    			{
			
				prevEigenVector[tid] = eigenVector[tid];

				__syncthreads();

        			float result = 0.0;
        			for (int j = 0; j < 3; j++) result += matrix[tid*3+j]*prevEigenVector[j];
            			
            			eigenVector[tid] = result;		
			
				__syncthreads();
			
			
				norm = sqrtf(eigenVector[0]*eigenVector[0] + eigenVector[1]*eigenVector[1] + eigenVector[2]*eigenVector[2]);
			
				eigenVector[tid] /= norm;
			
			
				__syncthreads();
			
			
				normDiff = sqrtf( (eigenVector[0]-prevEigenVector[0])*(eigenVector[0]-prevEigenVector[0]) +
               	        		   (eigenVector[1]-prevEigenVector[1])*(eigenVector[1]-prevEigenVector[1]) +
               	        		   (eigenVector[2]-prevEigenVector[2])*(eigenVector[2]-prevEigenVector[2]) );
			
				//printf("EigenVector: [%.4f, %.4f, %.4f] and difference is: %.4f\n", eigenVector[0], eigenVector[1], eigenVector[2], normDiff);	
		
			}
	
	
	    		// Calculate the eigenValue ???
    	    	
    			for (int i = 0; i < 3; i++)
        		lambda += matrix[tid * 3 + i] * eigenVector[i];
	
	
	
		}
	
	
		if (tid == 0){
	
			d_Polarity_Vec.x[rank] = eigenVector[0];
			d_Polarity_Vec.y[rank] = eigenVector[1];
			d_Polarity_Vec.z[rank] = eigenVector[2];
	
			if ( (step)%10 == 0 ){
			
				printf("step: %d\n",step);
			//	printf("Matrix\n");
			//	printf("%.4f, %.4f, %.4f\n", matrix[0], matrix[1], matrix[2]);
			//	printf("%.4f, %.4f, %.4f\n", matrix[3], matrix[4], matrix[5]);
			//	printf("%.4f, %.4f, %.4f\n", matrix[6], matrix[7], matrix[8]);
			
			//	printf("EigenVector: [%.4f, %.4f, %.4f]\n", eigenVector[0], eigenVector[1], eigenVector[2]);
    			
    			//	printf("\n");
			}
	
		}
	
	}


}				   
				   
				   
				   

#include <cuda.h>
#include <stdio.h>
#include "VectorFunctions.hpp"

__global__ void volumes( int No_of_C180s, int *C180_56,
                         float *X,    float *Y,   float *Z,
                         float *CMx , float *CMy, float *CMz, float *vol,
                         char* cell_div, float* d_DivisionVolume, bool checkSphericity,
                         float* areaList,
                         float stiffness1, bool useDifferentCell, float* d_younds_mod,
                         bool recalc_r0, float ApoVol, char* cell_Apo, float* d_ScaleFactor, int *num_cell_div, int *cell_div_inds){
                         
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
	

	if (volume < ApoVol){    	
	    cell_Apo[fullerene] = 1;	
	} else {
	    cell_Apo[fullerene] = 0;	
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
                               float* d_ScaleFactor,float* d_Youngs_mod, float* d_Growth_rate, float* d_DivisionVolume,
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
       	
       	if( d_CellINdex[rank] < 0 ){
       		d_CellINdex[newrank] = - NewCellInd;
       	} else {
       		d_CellINdex[newrank] = NewCellInd;
       	}
        
        } else {
        
              	d_Youngs_mod[newrank] = stiffness1; 
          	d_ScaleFactor[newrank] = 1;
          	d_Growth_rate[newrank] = rMax;
          	d_DivisionVolume[newrank] = divVol;
          	d_gamma_env[newrank] = gamma_visc;
          	d_viscotic_damp[newrank] = viscotic_damping;
          	d_CellINdex[newrank] = newrank;
          	  	
        }
        
        pressList[rank] = minPressure;             
	pressList[newrank] = minPressure;          
    
    }
    
    
}
    

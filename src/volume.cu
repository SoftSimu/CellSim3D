#include <cuda.h>
#include <stdio.h>
#include "VectorFunctions.hpp"
#include "SimParams.cuh"
#include "State.cuh"

__global__ void volumes(SimStatePtrs sim_state, sim_params_struct sim_params){
    
    __shared__ int* C180_56;
    __shared__ real* X;
    __shared__ real* Y;
    __shared__ real* Z;
    __shared__ real* CMx;
    __shared__ real* CMy;
    __shared__ real* CMz;
    __shared__ real* vol; 
    __shared__ char* cell_div;
    __shared__ real div_vol;
    __shared__ bool check_sphericity;
    __shared__ real* area_list;
    
    __shared__ float locX[192];
    __shared__ float locY[192];
    __shared__ float locZ[192];
    __shared__ float volume;
    __shared__ float area; 

    int fullerene = blockIdx.x;
    int tid       = threadIdx.x;
    
    if ( tid == 0){
        volume = 0.0f;
        area = 0.0f;
        C180_56 = sim_state.C180_56;
        X = sim_state.pos.x;
        Y = sim_state.pos.y;
        Z = sim_state.pos.z;

        CMx = sim_state.cellCOMs.x; 
        CMy = sim_state.cellCOMs.y; 
        CMz = sim_state.cellCOMs.z; 

        cell_div = sim_state.cellShouldDiv;

        div_vol = sim_params.core_params.division_vol;

        check_sphericity = sim_params.core_params.check_sphericity;
        area_list = sim_state.areas;
        vol = sim_state.vol; 
    }
    __syncthreads();

    if ( tid < 180 ){ 
        locX[tid] = X[192*fullerene+tid] -CMx[fullerene];
        locY[tid] = Y[192*fullerene+tid] -CMy[fullerene];
        locZ[tid] = Z[192*fullerene+tid] -CMz[fullerene];
    }

    __syncthreads();

    if ( tid < 92 ){

        float avX = 0.0f;
        float avY = 0.0f;
        float avZ = 0.0f;
#warning Some hardcoded parameters here...
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

            if (check_sphericity){
       
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
                faceArea += 0.5 * sqrt(xx*xx + yy*yy + zz*zz);
            }
        }
        atomicAdd(&volume, totvol);
    
        if (check_sphericity)
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
        }
        
        if (volume > div_vol){
            cell_div[fullerene] = 1; 
        }

        if (check_sphericity){
            area_list[fullerene] = area;
            float c = cbrtf(volume);
            float psi = 4.835975862049408 * c * c/area;
            if (abs(1.0f - psi) > 0.05 || psi < 0 || psi > 1){ // why 0.05?
                cell_div[fullerene] = 0;
                //printf("cell %d division rejected\n", fullerene);
            }
        }
    }
}

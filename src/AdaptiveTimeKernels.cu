#include "VectorFunctions.hpp"
#include "AdaptiveTimeKernels.cuh"
#include <cuda.h>
__global__ void Integrate(float *d_XP, float *d_YP, float *d_ZP,
                          float *d_X, float *d_Y, float *d_Z, 
                          float *d_XM, float *d_YM, float *d_ZM,
                          float *d_XMM, float *d_YMM, float *d_ZMM,
                          float *d_time, float mass,
                          float3 *d_forceList, int numCells, adp_coeffs a){
    
    const int cellInd = blockIdx.x;
    const int node = threadIdx.x; 
    const float delta_t = d_time[0]; 
    
    if (cellInd < numCells && node < 180){
        int nodeInd = cellInd*192 + node; 

        float3 posVecP = make_float3(d_XP[nodeInd], d_YP[nodeInd], d_ZP[nodeInd]);
        float3 posVec = make_float3(d_X[nodeInd], d_Y[nodeInd], d_Z[nodeInd]);
        float3 posVecM = make_float3(d_XM[nodeInd], d_YM[nodeInd], d_ZM[nodeInd]);
        float3 posVecMM = make_float3(d_XMM[nodeInd], d_YMM[nodeInd], d_ZMM[nodeInd]);

        posVecP =  -1*(a.k0*posVec + a.kn1*posVecM + a.kn2*posVecMM)/a.k1 + 
            ((delta_t*delta_t)/a.k1)*d_forceList[nodeInd];

        d_XP[nodeInd] = posVecP.x; 
        d_YP[nodeInd] = posVecP.y; 
        d_ZP[nodeInd] = posVecP.z;
    }
}

__global__ void ForwardTime(float *d_XP, float *d_YP, float *d_ZP,
                            float *d_X, float *d_Y, float *d_Z,
                            float *d_XM, float *d_YM, float *d_ZM, 
                            float *d_XMM, float *d_YMM, float *d_ZMM,
                            float *d_velListX, float *d_velListY, float *d_velListZ, 
                            int numCells, float *d_time){
    
    int nodeInd = blockIdx.x*blockDim.x + threadIdx.x;
    float delta_t = d_time[0]; 
    if (nodeInd < 192*numCells){
        
        d_velListX[nodeInd] = (d_XP[nodeInd] - d_X[nodeInd])/(delta_t); 
        d_velListY[nodeInd] = (d_YP[nodeInd] - d_Y[nodeInd])/(delta_t); 
        d_velListZ[nodeInd] = (d_ZP[nodeInd] - d_Z[nodeInd])/(delta_t);
        
        d_XMM[nodeInd] = d_XM[nodeInd]; 
        d_YMM[nodeInd] = d_YM[nodeInd]; 
        d_ZMM[nodeInd] = d_ZM[nodeInd];
    
    
        d_XM[nodeInd] = d_X[nodeInd]; 
        d_YM[nodeInd] = d_Y[nodeInd]; 
        d_ZM[nodeInd] = d_Z[nodeInd];
    
        d_X[nodeInd] = d_XP[nodeInd];
        d_Y[nodeInd] = d_YP[nodeInd];
        d_Z[nodeInd] = d_ZP[nodeInd];
    }
}

__global__ void ComputeTimeUpdate(float *d_XP, float *d_YP, float *d_ZP, 
                                  float *d_Xt, float *d_Yt, float *d_Zt,
                                  float *d_AdpErrors, float *d_time, float dt_max,
                                  float alpha, float beta, int numCells, float dt_tol){
    int index = blockIdx.x*blockDim.x + threadIdx.x;
    float delta_t = d_time[0]; 

    if (index < 192*numCells && threadIdx.x < 180){
        float3 Y1 = make_float3(d_XP[index], d_YP[index], d_ZP[index]);
        float3 Yt = make_float3(d_Xt[index], d_Yt[index], d_Zt[index]); 

        // ask bart about this error computation
        float e = abs(alpha/(beta-alpha))*mag(Yt-Y1);
        
        d_AdpErrors[index] = e;
        d_time[index] = min(dt_max, 0.9*delta_t*sqrt(dt_tol/e));
    }
}

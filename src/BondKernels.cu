#include "VectorFunctions.hpp"

__global__ void CalculateR0(float* d_R0, float* d_X, float* d_Y, float* d_Z,
                            int* d_C180_nn,
                            float* d_youngsModArray, float stiffness2, int No_of_C180s){

    int cellInd = blockIdx.x*blockDim.x + threadIdx.x;

    if (cellInd < No_of_C180s && d_youngsModArray[cellInd] == stiffness2){
        float avgR0 = 0.f;

        for (int node = 0; node < 180; ++node){
            int nodeInd = cellInd*192 + node;
            float3 nodePos = make_float3(d_X[nodeInd],
                                         d_Y[nodeInd],
                                         d_Z[nodeInd]);

            int n1 = d_C180_nn[0*192 + node];
            int n2 = d_C180_nn[1*192 + node];
            int n3 = d_C180_nn[2*192 + node];

            float3 n1Pos = make_float3(d_X[cellInd*192 + n1],
                                       d_Y[cellInd*192 + n1],
                                       d_Z[cellInd*192 + n1]);

            float3 n2Pos = make_float3(d_X[cellInd*192 + n2],
                                       d_Y[cellInd*192 + n2],
                                       d_Z[cellInd*192 + n2]);

            float3 n3Pos = make_float3(d_X[cellInd*192 + n3],
                                       d_Y[cellInd*192 + n3],
                                       d_Z[cellInd*192 + n3]);

            avgR0 += 1/3.f * (mag(nodePos - n1Pos) + mag(nodePos - n2Pos) + mag(nodePos - n3Pos));
        }
        
        d_R0[cellInd] = avgR0/180.f;
    }
    
}

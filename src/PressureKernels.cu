#include<cuda.h>
#include "postscript.h"

__global__ void PressureUpdate (float* d_pressList, float minPressure,
                                float maxPressure, float inc, int No_of_C180s){

    int cellInd = blockIdx.x*blockDim.x + threadIdx.x;
    float pressure = d_pressList[cellInd];
    
    if (cellInd < No_of_C180s){
        if (d_pressList[cellInd] < maxPressure){
            d_pressList[cellInd] += inc;
        }
    }
    
}


__global__ void PressureReset (int* d_resetIndices, float* d_pressList,
                               float pressureValue, int numCells){

    // d_resetIndices is an array of indeces to cells that have to have their
    // pressures reset

    int setInd = blockIdx.x*blockDim.x + threadIdx.x;
    if (setInd < numCells){
        // if (!d_resetIndices[setInd]){
        //     printf("WARNING:Trying to set pressure  of wrong cell\n");
        //     printf("Invalid cell index: %d setId: %d  numCells: %d", d_resetIndices[setInd], setInd, numCells); 
        // }

        d_pressList[d_resetIndices[setInd]] = pressureValue;
        d_resetIndices[setInd] = -1;
    }
}

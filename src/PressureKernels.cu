#include<cuda.h>
#include "postscript.h"

__global__ void PressureUpdate (float* d_pressList, float minPressure,
                                float maxPressure, float inc, int No_of_C180s,
                                bool useDifferentStiffnesses,
                                float stiffness1, float* d_youngs_mod, int step,
                                int phase_count){

    int cellInd = blockIdx.x*blockDim.x + threadIdx.x;
    

    if (cellInd < No_of_C180s){
        float pressure = d_pressList[cellInd];
        if (useDifferentStiffnesses == true){
            if (step < phase_count){
                if (d_pressList[cellInd] < maxPressure){
                    d_pressList[cellInd] += inc;
                }
            }
            else{
                if (d_youngs_mod[cellInd] == stiffness1){
                    if (d_pressList[cellInd] < maxPressure){
                        d_pressList[cellInd] += inc;
                    }
                }
                else {
                    if (d_pressList[cellInd] != 0.f){
                        d_pressList[cellInd] = 0.f; 
                    }
                }
                
            }
        }
        else {
            if (d_pressList[cellInd] < maxPressure){
                d_pressList[cellInd] += inc;
            }
        }
    }
}



__global__ void PressureReset (int* d_resetIndices, float* d_pressList,
                               float pressureValue, int numCells){

    // This kernel has really inefficient memory access
    // TODO: Investigate doing this on the host.


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

#ifndef STATE_CUH
#define STATE_CUH
#include "Types.cuh"
#include "SimParams.cuh"
#include "globals.cuh"
#include "VectorFunctions.hpp"

typedef struct{
    R3Nptrs posP;
    R3Nptrs pos;
    R3Nptrs posM;

    R3Nptrs vel; 
    R3Nptrs conForce;
    R3Nptrs disForce;
    R3Nptrs ranForce;
    R3Nptrs totForce;
    
    R3Nptrs cellCOMs;

    real* vol;
    real* areas;
    char* cellShouldDiv;
    real* boundingBoxes;
    real* pressures;
    real* sphericity;

    R3Nptrs mins; 
    R3Nptrs maxs; 

    int* nnList;
    int* numOfNNList;

    int* C180_nn;
    int* C180_sign;
    int* C180_56;

    int* resetIndices;

    real* bondStiffness;

    long int no_of_cells;
    long int no_new_cells; 

    real* R0;
    angles3* theta0;
} SimStatePtrs;


__device__ inline void write_to_R3N_state(R3Nptrs& state,
                                          const real3& value,
                                          const size_t globalInd){
    state.x[globalInd] = value.x;
    state.y[globalInd] = value.y;
    state.z[globalInd] = value.z;
}

__device__ inline real3 read_from_R3N_state(const R3Nptrs& state,
                                            const size_t globalInd){

    return make_real3(state.x[globalInd],
                      state.y[globalInd],
                      state.z[globalInd]);
}


__device__ inline void copy_R3N_state(R3Nptrs state,
                                       const size_t destGlobalInd, 
                                       const size_t sourceGlobalInd){
    
    state.x[destGlobalInd] = state.x[sourceGlobalInd];
    state.y[destGlobalInd] = state.y[sourceGlobalInd];
    state.z[destGlobalInd] = state.z[sourceGlobalInd];
}

#endif // STATE_CUH

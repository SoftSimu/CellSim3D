#include<cuda.h>
#include "Types.cuh"
#include "State.cuh"

// TODO:
// sort out the stiffness1 vs stiff_factor1 situation...

__global__ void PressureUpdate (SimStatePtrs sim_state, sim_params_struct sim_params,
                                long int step){
    
    __shared__ real *d_pressList;
    __shared__ real maxPressure;
    __shared__ real inc;
    __shared__ int num_of_cells;
    __shared__ real* d_youngs_mod;
    __shared__ long int phase_count;
    __shared__ bool useDifferentStiffnesses;
    __shared__ real stiffness1; 

    if (threadIdx.x == 0){
        d_pressList = sim_state.pressures;
        d_youngs_mod = sim_state.bondStiffness; 
        maxPressure = sim_params.core_params.max_pressure;
        inc = sim_params.core_params.growth_rate;
        num_of_cells = sim_state.no_of_cells;
        phase_count = sim_params.core_params.phase_count;
        useDifferentStiffnesses = sim_params.stiff_params.use_diff_stiff;
        stiffness1 = sim_params.core_params.stiff_factor1; 
    }
    __syncthreads();
        
    size_t cellInd = blockIdx.x*blockDim.x + threadIdx.x;

    if (cellInd < num_of_cells){
        real pressure = d_pressList[cellInd];
        if (useDifferentStiffnesses == true){
            if (step < phase_count){
                if (pressure < maxPressure){
                    pressure += inc;
                }
            }
            else{
                if (d_youngs_mod[cellInd] == stiffness1){
                    if (pressure < maxPressure){
                        pressure += inc;
                    }
                }
                else {
                    if (pressure != 0.f){
                        pressure = 0.f; 
                    }
                }
                
            }
        }
        else {
            if (pressure < maxPressure){
                pressure += inc;
            }
        }
        d_pressList[cellInd] = pressure; 
    }
}



__global__ void PressureReset (SimStatePtrs sim_state, sim_params_struct sim_params){

    // This kernel has really inefficient memory access
    // TODO: Investigate doing this on the host.

    __shared__ int *d_resetIndices;
    __shared__ real *d_pressList;
    __shared__ real pressureValue;
    __shared__ int no_new_cells;

    if (threadIdx.x == 0){
        d_resetIndices = sim_state.resetIndices;
        d_pressList = sim_state.pressures;
        pressureValue = sim_params.core_params.min_pressure;
        no_new_cells = sim_state.no_new_cells;
    }
    __syncthreads();
    
    // d_resetIndices is an array of indeces to cells that have to have their
    // pressures reset

    int setInd = blockIdx.x*blockDim.x + threadIdx.x;
    if (setInd < no_new_cells){
        // if (!d_resetIndices[setInd]){
        //     printf("WARNING:Trying to set pressure  of wrong cell\n");
        //     printf("Invalid cell index: %d setId: %d  numCells: %d", d_resetIndices[setInd], setInd, numCells);
        // }

        d_pressList[d_resetIndices[setInd]] = pressureValue;
        d_resetIndices[setInd] = -1;
    }
}

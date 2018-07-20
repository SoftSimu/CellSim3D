#include<cuda.h>
#include<stdio.h>
#include"VectorFunctions.hpp"
#include<curand_kernel.h>
#include"State.cuh"
#include"SimState.cuh"
cudaDeviceProp getDevice(void);

__global__ void  cell_division(long int pCellInd, long int daughtCellInd,
                               SimStatePtrs sim_state, sim_params_struct sim_params,
                               real3 norm);

__global__ void minmaxpre(SimStatePtrs sim_state);

__global__ void minmaxpost(SimStatePtrs sim_state);

__global__ void makeNNlist( SimStatePtrs sim_state, sim_params_struct sm);

__global__ void CenterOfMass(SimStatePtrs simState);

__global__ void volumes(SimStatePtrs sim_state, sim_params_struct sim_params);

int initialize_C180s(SimState& simState, const sim_params_struct& sim_params);
int InitializeRandom(int rand_seed);
int read_fullerene_nn(SimState& simState);

__global__ void CalculateConForce(SimStatePtrs sim_state, sim_params_struct sm);

__global__ void Integrate(SimStatePtrs sim_state, sim_params_struct sim_params);

__global__ void ForwardTime(SimStatePtrs sim_state);

__global__ void bounding_boxes(SimStatePtrs sim_state);

void rmarin(int ij, int kl);
void ranmar(float rvec[], int len);

__global__ void PressureUpdate (SimStatePtrs sim_state, sim_params_struct sim_params, long int step);

__global__ void PressureReset (SimStatePtrs sim_state, sim_params_struct sim_params);

__global__ void DeviceRandInit(curandState *rngState, uint *d_seeds, unsigned long long num);

__global__ void CalculateR0(float* d_R0, float* d_X, float* d_Y, float* d_Z,
                            int* d_C180_nn,
                            float* d_youngsModArray, float stiffness2, int No_of_C180s);

__global__ void CorrectCoMMotion(SimStatePtrs sim_state, real3 sysCM);

__global__ void VelocityUpdateA(SimStatePtrs sim_state, sim_params_struct sim_params);

__global__ void VelocityUpdateB(SimStatePtrs sim_state, sim_params_struct sim_params);

__global__ void CalculateDisForce(SimStatePtrs sim_state, sim_params_struct sim_params);

__global__ void SumForces(SimStatePtrs sim_state);

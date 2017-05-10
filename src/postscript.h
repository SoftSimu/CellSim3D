#include<cuda.h>
#include<stdio.h>
#include"VectorFunctions.hpp"
#include<curand_kernel.h>
#include"State.cuh"
cudaDeviceProp getDevice(void);

__global__ void  cell_division(int rank,
                               float* d_XP, float* d_YP, float* d_ZP,
                               float *d_X,  float *d_Y,  float *d_Z,
                               float* d_XM, float* d_YM, float* d_ZM,
                               float* AllCMx, float* AllCMy, float* AllCMz,
                               float* d_velListX, float* d_velListY, float* d_velListZ,
                               int No_of_C180s, float *d_randNorm, float repulsion_range);

__global__ void minmaxpre( int No_of_C180s, float *d_bounding_xyz,
                           float *Minx, float *Maxx, float *Miny, float *Maxy, float *Minz, float *Maxz);

__global__ void minmaxpost( int No_of_C180s,
                    float *Minx, float *Maxx, float *Miny, float *Maxy, float *Minz, float *Maxz);

__global__ void makeNNlist( int No_of_C180s, float *d_bounding_xyz,
                        float Minx, float Miny, float Minz, float attrac, int Xdiv, int Ydiv, int Zdiv,
                        int *d_NoofNNlist, int *d_NNlist, float DL);

__global__ void CenterOfMass(SimStatePtrs simState);

__global__ void volumes( int No_of_C180s, int *C180_56,
                         float *X,    float *Y,   float *Z,
                         float *CMx , float *CMy, float *CMz, float *vol,
                         char* cell_div, float divVol, bool checkSphericity,
                         float* areaList, int phase_count, int step,
                         float stiffness1, bool useDifferentStiffnesses, float* d_younds_mod,
                         bool recalc_r0);

int printboundingbox(int rank, float *bounding_xyz);
int initialize_C180s(int Orig_No_of_C180s);
int generate_random(int no_of_ran1_vectors);
int read_fullerene_nn(void);
int read_global_params(void);
int read_json_params(const char* inpFile);
__global__ void CalculateConForce( int No_of_C180s, int d_C180_nn[], int d_C180_sign[],
                           float d_X[],  float d_Y[],  float d_Z[],
                           float *d_CMx, float *d_CMy, float *d_CMz,
                           float* d_R0, float* d_pressList, float* d_Youngs_mod , float cellStiffness,
                           float internal_damping, const float *d_time,
                           float d_bounding_xyz[],
                           float attraction_strength, float attraction_range,
                           float repulsion_strength, float repulsion_range,
                           float viscotic_damping, float mass,
                           float Minx, float Miny,  float Minz, int Xdiv, int Ydiv, int Zdiv,
                           int *d_NoofNNlist, int *d_NNlist, float DL, float gamma_visc,
                           float wall1, float wall2,
                           float threshDist, bool useWalls,
                           float* d_velListX, float* d_velListY, float* d_velListZ,
                           bool useRigidSimulationBox, float boxLength, float* d_boxMin, float Youngs_mod,
                                bool constrainAngles, const angles3 d_theta0[], R3Nptrs d_forceList, float r_CM_o, float3 boxMax, R3Nptrs d_contactForces, const float* d_volList, const float div_vol);

__global__ void Integrate(float *d_XP, float *d_YP, float *d_ZP,
                          float *d_X, float *d_Y, float *d_Z,
                          float *d_XM, float *d_YM, float *d_ZM,
                          float *d_velListX, float *d_velListY, float *d_velListZ,
                          float *d_time, float mass,
                          R3Nptrs d_fConList, R3Nptrs d_fDisList, R3Nptrs d_fRanList,
                          int numCells, bool add_rands,
                          curandState *rngStates, float rand_scale_factor);

__global__ void ForwardTime(float *d_XP, float *d_YP, float *d_ZP,
                            float *d_X, float *d_Y, float *d_Z,
                            float *d_XM, float *d_YM, float *d_ZM,
                            int numCells);

__global__ void bounding_boxes( int No_of_C180s,
               float *d_XP, float *d_YP, float *d_ZP,
//               float *d_X,  float *d_Y,  float *d_Z,
//               float *d_XM, float *d_YM, float *d_ZM,
               float *bounding_xyz,
               float *avex, float *avey, float *avez);

void rmarin(int ij, int kl);
void ranmar(float rvec[], int len);



// Function to write the trajectory
void write_traj(int t_step, FILE* trajfile);
void write_vel(int t_step, FILE* velFile);
void WriteBinaryTraj(int t_step, FILE* trajfile, int frameCount);

// Function to get the indeces of dividing cells
inline void count_and_get_div();


inline void calc_sys_CM();

inline float getRmax2();

inline int num_cells_far();

__global__ void PressureUpdate (float* d_pressList, float minPressure,
                                float maxPressure, float inc, int No_of_C180s,
                                bool useDifferentStiffnesses, float stiffness1,
                                float* d_younds_mod, int step, int phase_count);

__global__ void PressureReset (int* d_resetIndices, float* d_pressList,
                               float minPressure, int numResetCells);

__global__ void CheckCellDivision(int No_of_C180s, int *C180_56,
                                  float *X,    float *Y,   float *Z,
                                  float *CMx , float *CMy, float *CMz, float *voll,
                                  int* d_C180_56,
                                  char* cell_div, float divVol, bool checkSpherecity);

__global__ void DeviceRandInit(curandState *rngState, uint *d_seeds, unsigned long long num);

__global__ void CalculateR0(float* d_R0, float* d_X, float* d_Y, float* d_Z,
                            int* d_C180_nn,
                            float* d_youngsModArray, float stiffness2, int No_of_C180s);

__global__ void CorrectCoMMotion(float* d_X, float* d_Y, float* d_Z,
                                 float sysCMx, float sysCMy, float sysCMz, long int numParts);

__global__ void VelocityUpdateA(float* d_VX, float* d_VY, float* d_VZ,
                                R3Nptrs fConList, R3Nptrs fRanList,
                                float dt, long int num_nodes, float m);

__global__ void VelocityUpdateB(float* d_VX, float* d_VY, float* d_VZ,
                                R3Nptrs fDisList, float dt, long int num_nodes, float m);

__global__ void CalculateDisForce( int No_of_C180s, int d_C180_nn[], int d_C180_sign[],
                                   float d_X[],  float d_Y[],  float d_Z[],
                                   float gamma_int,
                                   float d_bounding_xyz[],
                                   float attraction_range,
                                   float gamma_ext,
                                   float Minx, float Miny,  float Minz, int Xdiv, int Ydiv, int Zdiv,
                                   int *d_NoofNNlist, int *d_NNlist, float DL, float gamma_o,
                                   float* d_velListX, float* d_velListY, float* d_velListZ,
                                   R3Nptrs d_fDisList);

#include<cuda.h>
#include<stdio.h>
#include"VectorFunctions.hpp"
#include<curand_kernel.h>


//__constant__ float d_dt;

cudaDeviceProp getDevice(void);



__global__ void  cell_division(
                               float *d_X,  float *d_Y,  float *d_Z,
                               float* AllCMx, float* AllCMy, float* AllCMz,
                               float* d_velListX, float* d_velListY, float* d_velListZ, 
                               int No_of_C180s, float repulsion_range, float* d_asym,
                               bool useDifferentCell, bool daughtSame,
                               int NewCellInd, float stiffness1, float rMax, float divVol, float gamma_visc, float viscotic_damping,
                               float* d_ScaleFactor,float* d_Youngs_mod, float* d_Growth_rate, float* d_DivisionVolume,
                               float* d_gamma_env, float* d_viscotic_damp, int* d_CellINdex,
                               R3Nptrs d_DivPlane, int *num_cell_div, int *cell_div_inds, float *pressList, float minPressure);



__global__ void makeNNlistPin( int impurityNum, float *CMx, float *CMy,float *CMz,
                           int Xdiv, int Ydiv, int Zdiv, float3 BoxMin,
                           int *d_NoofNNlistPin, int *d_NNlistPin, float DL);

__global__ void makeNNlistPBCPin(int impurityNum, float *CMx, float *CMy,float *CMz,
                           float attrac, int Xdiv, int Ydiv, int Zdiv, float3 boxMax,
                           int *d_NoofNNlistPin, int *d_NNlistPin, float3 DLp, bool useRigidBoxZ,bool useRigidBoxY);

                           
__global__ void makeNNlistLEbcPin(int impurityNum, float *CMx, float *CMy,float *CMz,
                           float attrac, int Xdiv, int Ydiv, int Zdiv, float3 boxMax,
                           int *d_NoofNNlistPin, int *d_NNlistPin, float3 DLp, float Pshift,bool useRigidBoxZ);


__global__ void makeNNlist( int No_of_C180s, float *CMx, float *CMy,float *CMz, float *CMxNNlist, float *CMyNNlist, float *CMzNNlist,
                           int Xdiv, int Ydiv, int Zdiv, float3 BoxMin,
                           int *d_NoofNNlist, int *d_NNlist, float DL);

__global__ void makeNNlistPBC(int No_of_C180s, float *CMx, float *CMy,float *CMz, float *CMxNNlist, float *CMyNNlist, float *CMzNNlist,
                           float attrac, int Xdiv, int Ydiv, int Zdiv, float3 boxMax,
                           int *d_NoofNNlist, int *d_NNlist, float3 DLp, bool useRigidBoxZ,bool useRigidBoxY);

                           
__global__ void makeNNlistLEbc(int No_of_C180s, float *CMx, float *CMy,float *CMz, float *CMxNNlist, float *CMyNNlist, float *CMzNNlist,
                           float attrac, int Xdiv, int Ydiv, int Zdiv, float3 boxMax,
                           int *d_NoofNNlist, int *d_NNlist, float3 DLp, float Pshift,bool useRigidBoxZ);


__global__ void CenterOfMass( int No_of_C180s,
               float *d_XP, float *d_YP, float *d_ZP,
               float *CMx, float *CMy, float *CMz);
               
__global__ void VelocityCenterOfMass( int No_of_C180s, 
               float *d_velListX, float *d_velListY, float *d_velListZ, 
               float *d_VCMx, float *d_VCMy,float *d_VCMz);
               

__global__ void volumes( int No_of_C180s, int *C180_56,
                         float *X,    float *Y,   float *Z,
                         float *CMx , float *CMy, float *CMz, float *vol,
                         char* cell_div, float* d_DivisionVolume, bool checkSphericity,
                         float* areaList, 
                         float stiffness1, bool useDifferentCell, float* d_younds_mod,
                         bool recalc_r0,float ApoVol ,char* d_cell_Apo,float* d_ScaleFactor, int *num_cell_div, int *cell_div_inds);

int printboundingbox(int rank, float *bounding_xyz);
int initialize_C180s(int Orig_No_of_C180s, int impurityNum);
int initialize_Vel(int Orig_No_of_C180s);
int generate_random(int no_of_ran1_vectors);
int read_fullerene_nn(void);
int read_global_params(void);
int read_json_params(const char* inpFile);
int PSSETUP(FILE* outfile);
int PSLINE(float X1, float Y1, float X2, float Y2, FILE *outfile);
int PSCIRCLE(float X,float Y,FILE *outfile);
int PSNET(int NN,int sl,float L1, float *X, float *Y, float *Z, int CCI[2][271]);

int PSNUM(float X, float Y, int NUMBER, FILE *outfile);


__global__ void CalculateConForce( int No_of_C180s, int d_C180_nn[], int d_C180_sign[],
                           float d_X[],  float d_Y[],  float d_Z[],
                           float *d_CMx, float *d_CMy, float *d_CMz,
                           float d_XPin[],  float d_YPin[],  float d_ZPin[],
                           float *d_CMxPin, float *d_CMyPin, float *d_CMzPin,
                           float* d_R0, float* d_ScaleFactor, float* d_pressList, float* d_stiffness, 
                           float attraction_strength, float attraction_range,
                           float repulsion_strength, float repulsion_range,
                           float* d_viscotic_damp,
                           int Xdiv, int Ydiv, int Zdiv, float3 boxMax,
                           int *d_NoofNNlist, int *d_NNlist, int *d_NoofNNlistPin, int *d_NNlistPin,  float DL, float* d_gamma_env,
                           float threshDist, 
                           float3 BoxMin, float Youngs_mod, 
                           bool constrainAngles, const angles3 d_theta0[], R3Nptrs d_forceList, R3Nptrs d_ExtForces, 
                           bool impurity, float f_range);

                           

__global__ void CalculateConForcePBC( int No_of_C180s, int d_C180_nn[], int d_C180_sign[],
                           float d_X[],  float d_Y[],  float d_Z[],
                           float *d_CMx, float *d_CMy, float *d_CMz,
                           float d_XPin[],  float d_YPin[],  float d_ZPin[],
                           float *d_CMxPin, float *d_CMyPin, float *d_CMzPin,
                           float* d_R0,float* d_ScaleFactor, float* d_pressList, float* d_stiffness, 
                           float attraction_strength, float attraction_range,
                           float repulsion_strength, float repulsion_range,
                           float* d_viscotic_damp,
                           int Xdiv, int Ydiv, int Zdiv, float3 boxMax,
                           int *d_NoofNNlist, int *d_NNlist, int *d_NoofNNlistPin, int *d_NNlistPin, float3 DLp, float* d_gamma_env,
                           float threshDist, 
                           float3 BoxMin, float Youngs_mod, 
                           bool constrainAngles, const angles3 d_theta0[], R3Nptrs d_forceList,
                           bool useRigidBoxZ, bool useRigidBoxY, bool impurity, float f_range);
                           
__global__ void CalculateConForceLEbc( int No_of_C180s, int d_C180_nn[], int d_C180_sign[],
                           float d_X[],  float d_Y[],  float d_Z[],
                           float *d_CMx, float *d_CMy, float *d_CMz,
                           float d_XPin[],  float d_YPin[],  float d_ZPin[],
                           float *d_CMxPin, float *d_CMyPin, float *d_CMzPin,
                           float* d_R0, float* d_ScaleFactor, float* d_pressList, float* d_stiffness, 
                           float attraction_strength, float attraction_range,
                           float repulsion_strength, float repulsion_range,
                           float* d_viscotic_damp,
                           int Xdiv, int Ydiv, int Zdiv, float3 boxMax,
                           int *d_NoofNNlist, int *d_NNlist, int *d_NoofNNlistPin, int *d_NNlistPin, float3 DLp, float* d_gamma_env,
                           float threshDist,  
                           float3 BoxMin, float Youngs_mod, 
                           bool constrainAngles, const angles3 d_theta0[], R3Nptrs d_forceList,
                           float Pshift , bool useRigidBoxZ, bool impurity, float f_range);

__global__ void Integrate(float *d_X, float *d_Y, float *d_Z,
                          float *d_velListX, float *d_velListY, float *d_velListZ,
                          float dt, float m,
                          R3Nptrs d_fConList, R3Nptrs d_fDisList, R3Nptrs d_fRanList,
                          int numCells);

__global__ void ForwardTime(float *d_XP, float *d_YP, float *d_ZP,
                            float *d_X, float *d_Y, float *d_Z,
                            int numCells);


void rmarin(int ij, int kl);
void ranmar(float rvec[], int len);



// Function to write the trajectory
void write_traj(int t_step, FILE* trajfile);
void write_vel(int t_step, FILE* velFile,int frameCount);
void WriteBinaryTraj(int t_step, FILE* trajfile, int frameCount);
void WriteBinaryTrajPin(int t_step, FILE* trajFile, int frameCount);
void write_trajPin(int t_step, FILE* trajfile);

// read and write restart function
int writeRestartFile(int t_step, int frameCount);
int ReadRestartFile();


// Function to get the indeces of dividing cells
inline void count_and_get_div();
inline void count_and_die();
inline void initialize_Plane( int MaxNoofC180s);


int DispersityFunc(int Orig_No_of_C180s);
int SecondCell (int Orig_No_of_C180s);
void ShapeScaler (float* initx,float* inity,float* initz);
void RotationMatrix(float* RMat,float* axis,float* theta);
inline void calc_sys_CM();
inline void calc_sys_VCM();


inline float getRmax2();

inline int num_cells_far();

__global__ void PressureUpdate (float* d_pressList,
                                float maxPressure, float* d_Growth_rate, int No_of_C180s,
                                float* d_younds_mod);

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

void writeForces(FILE* forceFile, int t_step, int num_cells);

__global__ void CorrectCoMMotion(float* d_X, float* d_Y, float* d_Z,
                                 R3Nptrs d_sysCM, float3 BoxCen, long int numParts);

__global__ void CorrectCoMVelocity(float* d_velListX, float* d_velListY, float* d_velListZ,
                                   R3Nptrs d_sysVCM, long int numParts);

__global__ void SysCMpost( int No_of_C180s, float *d_Cx, float *d_Cy,float *d_Cz, 
			   float* SysCx, float* SysCy, float* SysCz);

__global__ void SysCM( int No_of_C180s, int reductionblocks,
			float* SysCx, float* SysCy, float* SysCz,
			R3Nptrs d_sysCM);


__global__ void VelocityUpdateA(float* d_VX, float* d_VY, float* d_VZ,
                                R3Nptrs fConList, R3Nptrs fRanList,
                                float dt, long int num_nodes, float m );

__global__ void VelocityUpdateB(float* d_VX, float* d_VY, float* d_VZ,
                                R3Nptrs fDisList, float dt, long int num_nodes, float m );

__global__ void CalculateDisForce(int No_of_C180s, int d_C180_nn[], int d_C180_sign[],
                                   float d_X[],  float d_Y[],  float d_Z[],
                                   float *d_CMx, float *d_CMy, float *d_CMz,
                                   float d_XPin[],  float d_YPin[],  float d_ZPin[],
                                   float *d_CMxPin, float *d_CMyPin, float *d_CMzPin,
                                   float gamma_int,
                                   float attraction_range,
                                   float* d_viscotic_damp,
                                   int Xdiv, int Ydiv, int Zdiv,  float3 BoxMin,
                                   int *d_NoofNNlist, int *d_NNlist, int *d_NoofNNlistPin, int *d_NNlistPin, float DL, float* d_gamma_env,
                                   float* d_velListX, float* d_velListY, float* d_velListZ,
                                   R3Nptrs d_fDisList, bool impurity, float f_range);
                                   

__global__ void CalculateDisForcePBC( int No_of_C180s, int d_C180_nn[], int d_C180_sign[],
                                   float d_X[],  float d_Y[],  float d_Z[],
                                   float *d_CMx, float *d_CMy, float *d_CMz,
                                   float d_XPin[],  float d_YPin[],  float d_ZPin[],
                                   float *d_CMxPin, float *d_CMyPin, float *d_CMzPin,
                                   float gamma_int,
                                   float attraction_range,
                                   float* d_viscotic_damp,
                                   int Xdiv, int Ydiv, int Zdiv,float3 boxMax,
                                   int *d_NoofNNlist, int *d_NNlist, int *d_NoofNNlistPin, int *d_NNlistPin, float3 DLp, float* d_gamma_env,
                                   float* d_velListX, float* d_velListY, float* d_velListZ,
                                   R3Nptrs d_fDisList, bool useRigidBoxZ, bool useRigidBoxY, bool impurity, float f_range);
                                   
__global__ void CalculateDisForceLEbc( int No_of_C180s, int d_C180_nn[], int d_C180_sign[],
                                   float d_X[],  float d_Y[],  float d_Z[],
                                   float *d_CMx, float *d_CMy, float *d_CMz,
                                   float d_XPin[],  float d_YPin[],  float d_ZPin[],
                                   float *d_CMxPin, float *d_CMyPin, float *d_CMzPin,
                                   float gamma_int,
                                   float attraction_range,
                                   float* d_viscotic_damp,
                                   int Xdiv, int Ydiv, int Zdiv,float3 boxMax,
                                   int *d_NoofNNlist, int *d_NNlist, int *d_NoofNNlistPin, int *d_NNlistPin, float3 DLp, float* d_gamma_env,
                                   float* d_velListX, float* d_velListY, float* d_velListZ,
                                   R3Nptrs d_fDisList,float Pshift, float Vshift ,bool useRigidBoxZ, bool impurity, float f_range);

__global__ void CalculateRanForce(int No_of_C180s, curandState *d_rngStates, float rand_scale_factor,
                                  R3Nptrs d_fRanList);


__global__ void CoorUpdatePBC (float *d_X, float *d_Y, float *d_Z, 
                               float *d_CMx, float *d_CMy, float *d_CMz,
                               float3 boxMax, float divVol,int numCells,
                               bool useRigidBoxZ, bool useRigidBoxY );
                               
__global__ void UpdateLEbc (float *d_X, float *d_Y, float *d_Z, 
                               float* d_VX, float* d_VY, float* d_VZ,
                               float *d_CMx, float *d_CMy, float *d_CMz,
                               float3 boxMax, float divVol, int numCells,
                               float Pshift, float Vshift,bool useRigidBoxZ );
                               
__global__ void ShiftInf (float* d_X,float* d_Y,float* d_Z,
                              float* d_XM,float* d_YM,float* d_ZM,
                              float* d_velListX,float* d_velListY,float* d_velListZ,
                              float* d_pressList,float* d_Youngs_mod, float* d_Growth_rate,
                              int* d_CellINdex,int No_of_C180s,int Aporank);


__global__ void DangerousParticlesFinder(int No_of_C180s, float *CMx, float *CMy,float *CMz,
					  float *CMxNNlist, float *CMyNNlist, float *CMzNNlist,
					  float BufferDistance, int *num_cell_dang, int* cell_dang_inds, char* cell_dang,
					  float3 boxMax);

                              
__global__ void DangerousParticlesFinderPBC(int No_of_C180s, float *CMx, float *CMy,float *CMz,
					  float *CMxNNlist, float *CMyNNlist, float *CMzNNlist,
					  float BufferDistance, int *num_cell_dang, int* cell_dang_inds, char* cell_dang,
					  float3 boxMax, bool useRigidBoxZ, bool useRigidBoxY);
					  
                              
__global__ void DangerousParticlesFinderLEbc(int No_of_C180s, float *CMx, float *CMy,float *CMz,
					  float *CMxNNlist, float *CMyNNlist, float *CMzNNlist,
					  float BufferDistance, int *num_cell_dang, int* cell_dang_inds, char* cell_dang,
					  float3 boxMax, bool useRigidBoxZ, bool useRigidBoxY);                              

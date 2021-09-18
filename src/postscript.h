#include "mpi.h"
#include<cuda.h>
#include<stdio.h>
#include"VectorFunctions.hpp"
#include<curand_kernel.h>


//__constant__ float d_dt;

cudaDeviceProp getDevice(int idev);



__global__ void  cell_division(float *d_X,  float *d_Y,  float *d_Z,
                               float* AllCMx, float* AllCMy, float* AllCMz,
                               float* d_velListX, float* d_velListY, float* d_velListZ, 
                               int No_of_C180s, float repulsion_range, float* d_asym,
                               bool useDifferentCell, bool daughtSame,
                               int NewCellInd, float stiffness1, float rMax, float divVol, float gamma_visc, float viscotic_damping,
                               float squeeze_rate1, float Apo_rate1,
                               float* d_ScaleFactor,float* d_Youngs_mod, float* d_Growth_rate, float* d_DivisionVolume,
                               float* d_squeeze_rate, float* d_Apo_rate,
                               float* d_gamma_env, float* d_viscotic_damp, int* d_CellINdex,
				R3Nptrs d_DivPlane, int *num_cell_div, int *cell_div_inds, float *pressList, float minPressure);


__global__ void makeNNlist(int No_of_C180s, float *CMx, float *CMy,float *CMz, float *CMxNNlist, float *CMyNNlist, float *CMzNNlist,
                           int Xdiv, int Ydiv, int Zdiv, float3 BoxMin,
                           int *d_NoofNNlist, int *d_NNlist, float DL);

__global__ void makeNNlistPin(int impurityNum, float *CMx, float *CMy,float *CMz,
                           int Xdiv, int Ydiv, int Zdiv, float3 Subdivision_min,
                           int *d_NoofNNlistPin, int *d_NNlistPin, float DL);

__global__ void makeNNlistPBCPin(int impurityNum, float *CMx, float *CMy,float *CMz,
                           float attrac, int Xdiv, int Ydiv, int Zdiv, float3 Subdivision_min,
                           int *d_NoofNNlistPin, int *d_NNlistPin, float DL, bool useRigidBoxZ, bool useRigidBoxY, bool useRigidBoxX);

                           
__global__ void makeNNlistLEbcPin(int impurityNum, float *CMx, float *CMy,float *CMz,
                           float attrac, int Xdiv, int Ydiv, int Zdiv, float3 boxMax,
                           int *d_NoofNNlistPin, int *d_NNlistPin, float3 DLp, float Pshift,bool useRigidBoxZ, bool useRigidBoxY, bool useRigidBoxX);


__global__ void makeNNlistMultiGpu(int No_of_C180s, float R_ghost_buffer, float *d_CMx, float *d_CMy,float *d_CMz, float *CMxNNlist, float *CMyNNlist, float *CMzNNlist,
                           int Xdiv, int Ydiv, int Zdiv, float3 Subdivision_min, float3 Subdivision_max, float3 BoxMin, float3 boxMax,
                           int *d_NoofNNlist, int *d_NNlist, float DL, int* d_counter_gc_e, int* d_counter_gc_w,
                           int* d_counter_gc_n, int* d_counter_gc_s, int* d_counter_gc_u, int* d_counter_gc_d,
                           int* d_Ghost_Cells_ind_EAST, int* d_Ghost_Cells_ind_WEST, int* d_Ghost_Cells_ind_NORTH, int* d_Ghost_Cells_ind_SOUTH,
                           int* d_Ghost_Cells_ind_UP, int* d_Ghost_Cells_ind_DOWN);

__global__ void makeNNlistPBC(int No_of_C180s, float *CMx, float *CMy,float *CMz, float *CMxNNlist, float *CMyNNlist, float *CMzNNlist,
                           float attrac, int Xdiv, int Ydiv, int Zdiv, float3 boxMax,
                           int *d_NoofNNlist, int *d_NNlist, float3 DLp, bool useRigidBoxZ,bool useRigidBoxY);

__global__ void makeNNlistMultiGpuPBC( int No_of_C180s, float R_ghost_buffer, float *d_CMx, float *d_CMy,float *d_CMz, float *CMxNNlist, float *CMyNNlist, float *CMzNNlist,
                           int Xdiv, int Ydiv, int Zdiv, float3 Subdivision_min, float3 Subdivision_max, float3 BoxMin, float3 boxMax,
                           int *d_NoofNNlist, int *d_NNlist, float DL, int* d_counter_gc_e, int* d_counter_gc_w,
                           int* d_counter_gc_n, int* d_counter_gc_s, int* d_counter_gc_u, int* d_counter_gc_d,
                           int* d_Ghost_Cells_ind_EAST, int* d_Ghost_Cells_ind_WEST, int* d_Ghost_Cells_ind_NORTH, int* d_Ghost_Cells_ind_SOUTH,
                           int* d_Ghost_Cells_ind_UP, int* d_Ghost_Cells_ind_DOWN );
                           
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
                         float stiffness1, bool useDifferentCell, float* d_younds_mod, float* d_Growth_rate,
                         bool recalc_r0, float ApoVol, float* d_ScaleFactor,
                         int *num_cell_div, int *cell_div_inds, char* d_cell_Apo, int* d_num_cell_Apo, int *d_cell_Apo_inds);

int printboundingbox(int rank, float *bounding_xyz);
int initialize_C180s(int* Orig_No_of_C180s, int* impurityNum);
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
                           float3 BoxMin, float3 Subdivision_min, float Youngs_mod, 
                           bool constrainAngles, const angles3 d_theta0[], R3Nptrs d_forceList, R3Nptrs d_ExtForces, 
                           bool impurity, float f_range,
                           bool useRigidSimulationBox, bool useRigidBoxZ, bool useRigidBoxY, bool useRigidBoxX);

                           

__global__ void CalculateConForcePBC( int No_of_C180s, int d_C180_nn[], int d_C180_sign[],
                           float d_X[],  float d_Y[],  float d_Z[],
                           float *d_CMx, float *d_CMy, float *d_CMz,
                           float d_XPin[],  float d_YPin[],  float d_ZPin[],
                           float *d_CMxPin, float *d_CMyPin, float *d_CMzPin,
                           float* d_R0,float* d_ScaleFactor, float* d_pressList, float* d_stiffness, 
                           float attraction_strength, float attraction_range,
                           float repulsion_strength, float repulsion_range,
                           float* d_viscotic_damp,
                           int Xdiv, int Ydiv, int Zdiv, float3 boxMax, float3 Subdivision_min,
                           int *d_NoofNNlist, int *d_NNlist, int *d_NoofNNlistPin, int *d_NNlistPin, float3 DLp, float* d_gamma_env,
                           float threshDist, 
                           float3 BoxMin, float Youngs_mod, 
                           bool constrainAngles, const angles3 d_theta0[], R3Nptrs d_forceList, R3Nptrs d_ExtForces, 
                           bool useRigidBoxZ, bool useRigidBoxY, bool impurity, float f_range);

__global__ void CalculateConForceMultiGPUPBC( int No_of_C180s, int d_C180_nn[], int d_C180_sign[],
                           	float d_X[],  float d_Y[],  float d_Z[],
                           	float *d_CMx, float *d_CMy, float *d_CMz,
                           	float d_XPin[],  float d_YPin[],  float d_ZPin[],
                           	float *d_CMxPin, float *d_CMyPin, float *d_CMzPin,                   
                           	float* d_R0,float* d_ScaleFactor, float* d_pressList, float* d_stiffness, 
                           	float attraction_strength, float attraction_range,
                           	float repulsion_strength, float repulsion_range,
                           	float* d_viscotic_damp,
                           	int Xdiv, int Ydiv, int Zdiv,float3 boxMax, float3 Subdivision_min,
                           	int *d_NoofNNlist, int *d_NNlist, int *d_NoofNNlistPin, int *d_NNlistPin, float DL, float* d_gamma_env,
                           	float threshDist, 
                           	float3 BoxMin, float Youngs_mod, 
                           	bool constrainAngles, const angles3 d_theta0[], R3Nptrs d_forceList, R3Nptrs d_ExtForces,
                           	bool useRigidBoxZ, bool useRigidBoxY, bool useRigidBoxX, bool impurity, float f_range);
                           
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
void WriteBinaryTraj(int t_step, FILE* trajfile, int frameCount,int rank);
void WriteBinaryTrajPin(int t_step, FILE* trajFile, int frameCount);
void write_trajPin(int t_step, FILE* trajfile);

// read and write restart function
int writeRestartFile(int t_step, int frameCount);
int ReadRestartFile();
int ReadPinFile();


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
                                   int Xdiv, int Ydiv, int Zdiv, float3 Subdivision_min,
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
                                   int Xdiv, int Ydiv, int Zdiv,float3 boxMax, float3 Subdivision_min,
                                   int *d_NoofNNlist, int *d_NNlist, int *d_NoofNNlistPin, int *d_NNlistPin, float3 DLp, float* d_gamma_env,
                                   float* d_velListX, float* d_velListY, float* d_velListZ,
                                   R3Nptrs d_fDisList, bool useRigidBoxZ, bool useRigidBoxY, bool impurity, float f_range);


__global__ void CalculateDisForceMultiGPUPBC( int No_of_C180s, int d_C180_nn[], int d_C180_sign[],
                                   	float d_X[],  float d_Y[],  float d_Z[],
                                   	float *d_CMx, float *d_CMy, float *d_CMz,
                           	    	float d_XPin[],  float d_YPin[],  float d_ZPin[],
                           	    	float *d_CMxPin, float *d_CMyPin, float *d_CMzPin,                                   
                                   	float gamma_int,
                                   	float attraction_range,
                                   	float* d_viscotic_damp,
                                   	int Xdiv, int Ydiv, int Zdiv,float3 boxMax, float3 Subdivision_min,
                                   	int *d_NoofNNlist, int *d_NNlist, int *d_NoofNNlistPin, int *d_NNlistPin, float DL, float* d_gamma_env,
                                   	float* d_velListX, float* d_velListY, float* d_velListZ,
                                   	R3Nptrs d_fDisList, bool impurity, float f_range);
                                   
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
					  float BufferDistance, int *num_cell_dang, int* cell_dang_inds, char* cell_dang);

                              
__global__ void DangerousParticlesFinderPBC(int No_of_C180s, float *CMx, float *CMy,float *CMz,
					  float *CMxNNlist, float *CMyNNlist, float *CMzNNlist,
					  float BufferDistance, int *num_cell_dang, int* cell_dang_inds, char* cell_dang,
					  float3 boxMax, bool useRigidBoxZ, bool useRigidBoxY, bool useRigidBoxX);
					  
                              
__global__ void DangerousParticlesFinderLEbc(int No_of_C180s, float *CMx, float *CMy,float *CMz,
					  float *CMxNNlist, float *CMyNNlist, float *CMzNNlist,
					  float BufferDistance, int *num_cell_dang, int* cell_dang_inds, char* cell_dang,
					  float3 boxMax, bool useRigidBoxZ, bool useRigidBoxY); 

__global__ void UpdateNNlistDivision(int No_of_C180s, int non_divided_cells, float *CMx, float *CMy,float *CMz,
					float *CMxNNlist, float *CMyNNlist, float *CMzNNlist,
                           		int Xdiv, int Ydiv, int Zdiv, float3 BoxMin,
                           		int *d_NoofNNlist, int *d_NNlist, float DL); 

                           	
__global__ void UpdateNNlistDivisionPBC(int No_of_C180s, int non_divided_cells, float *CMx, float *CMy,float *CMz, float *CMxNNlist, float *CMyNNlist, float *CMzNNlist,
                           		 int Xdiv, int Ydiv, int Zdiv, float3 boxMax,
                           		 int *d_NoofNNlist, int *d_NNlist, float3 DLp, bool useRigidBoxZ, bool useRigidBoxY);


__global__ void UpdateNNlistDivisionLEbc(int No_of_C180s, int non_divided_cells, float *CMx, float *CMy,float *CMz, float *CMxNNlist, float *CMyNNlist, float *CMzNNlist,
                           		   int Xdiv, int Ydiv, int Zdiv, float3 boxMax,
                           		   int *d_NoofNNlist, int *d_NNlist, float3 DLp, float Pshift,bool useRigidBoxZ);   

__global__ void Cell_removing (int No_of_C180s, int num_cell_Apo, int* d_counter,
				float *d_X,  float *d_Y,  float *d_Z,
                               float* d_velListX, float* d_velListY, float* d_velListZ, 
                               float* d_ScaleFactor,float* d_Youngs_mod, float* d_Growth_rate, float* d_DivisionVolume,
                               float* d_gamma_env, float* d_viscotic_damp,float* d_pressList, int* d_CellINdex,
                               float* Apo_rate, float* squeeze_rate,
				int* d_cell_Apo_inds, char* cell_Apo);

__global__ void CellApoptosis(int No_of_C180s, curandState *d_rngStatesApo, float* d_Apo_rate,
 				float* d_Growth_rate, float* d_squeeze_rate, int* d_Num_shrink_Cell);
 				

__global__ void ghost_cells_finder_Auxiliary(int No_of_C180s, int All_Cells, float *d_CM , 
						float Sub_max, float Sub_min, float R_ghost_buffer,
						int* d_counter_gc_r, int* d_counter_gc_l,
                         			int* d_Ghost_Cells_ind_R, int* d_Ghost_Cells_ind_L);

__global__ void Ghost_Cells_Pack(int No_of_Ghost_cells_buffer, int* d_Ghost_Cells_ind,
				float *d_X,  float *d_Y,  float *d_Z,
                               float* d_velListX, float* d_velListY, float* d_velListZ,
                               float* d_CMx, float* d_CMy, float* d_CMz,
				float *d_X_gc_buffer,  float *d_Y_gc_buffer,  float *d_Z_gc_buffer,
                              float* d_velListX_gc_buffer, float* d_velListY_gc_buffer, float* d_velListZ_gc_buffer,
                              float* d_CMx_gc_buffer, float* d_CMy_gc_buffer, float* d_CMz_gc_buffer);

__global__ void Ghost_Cells_Pack_PBC_X(int No_of_Ghost_cells_buffer, int No_of_Ghost_cells_buffer_R, int* d_Ghost_Cells_ind, float3 boxMax, float R_ghost_buffer,
					float *d_X,  float *d_Y,  float *d_Z,
                               	float* d_velListX, float* d_velListY, float* d_velListZ,
                               	float* d_CMx, float* d_CMy, float* d_CMz,
					float *d_X_gc_buffer,  float *d_Y_gc_buffer,  float *d_Z_gc_buffer,
                              	float* d_velListX_gc_buffer, float* d_velListY_gc_buffer, float* d_velListZ_gc_buffer,
                              	float* d_CMx_gc_buffer, float* d_CMy_gc_buffer, float* d_CMz_gc_buffer);
                            
__global__ void Ghost_Cells_Pack_PBC_Y(int No_of_Ghost_cells_buffer, int No_of_Ghost_cells_buffer_R, int* d_Ghost_Cells_ind, float3 boxMax, float R_ghost_buffer,
					float *d_X,  float *d_Y,  float *d_Z,
                               	float* d_velListX, float* d_velListY, float* d_velListZ,
                               	float* d_CMx, float* d_CMy, float* d_CMz,
					float *d_X_gc_buffer,  float *d_Y_gc_buffer,  float *d_Z_gc_buffer,
                              	float* d_velListX_gc_buffer, float* d_velListY_gc_buffer, float* d_velListZ_gc_buffer,
                              	float* d_CMx_gc_buffer, float* d_CMy_gc_buffer, float* d_CMz_gc_buffer);
                              	
__global__ void Ghost_Cells_Pack_PBC_Z(int No_of_Ghost_cells_buffer, int No_of_Ghost_cells_buffer_R, int* d_Ghost_Cells_ind, float3 boxMax, float R_ghost_buffer,
					float *d_X,  float *d_Y,  float *d_Z,
                               	float* d_velListX, float* d_velListY, float* d_velListZ,
                               	float* d_CMx, float* d_CMy, float* d_CMz,
					float *d_X_gc_buffer,  float *d_Y_gc_buffer,  float *d_Z_gc_buffer,
                              	float* d_velListX_gc_buffer, float* d_velListY_gc_buffer, float* d_velListZ_gc_buffer,
                              	float* d_CMx_gc_buffer, float* d_CMy_gc_buffer, float* d_CMz_gc_buffer);

__global__ void Ghost_Cells_Pack_LEbc_X(int No_of_Ghost_cells_buffer, int No_of_Ghost_cells_buffer_R, int* d_Ghost_Cells_ind, float3 boxMax, float R_ghost_buffer,
					float Pshift, float Vshift,
					float *d_X,  float *d_Y,  float *d_Z,
                               	float* d_velListX, float* d_velListY, float* d_velListZ,
                               	float* d_CMx, float* d_CMy, float* d_CMz,
					float *d_X_gc_buffer,  float *d_Y_gc_buffer,  float *d_Z_gc_buffer,
                              	float* d_velListX_gc_buffer, float* d_velListY_gc_buffer, float* d_velListZ_gc_buffer,
                              	float* d_CMx_gc_buffer, float* d_CMy_gc_buffer, float* d_CMz_gc_buffer);
                              
__global__ void UpdateNNlistWithGhostCells(int No_of_C180s, int All_Cells, float *d_CMx_gc, float *d_CMy_gc,float *d_CMz_gc,
                           int Xdiv, int Ydiv, int Zdiv, float3 Subdivision_min,
                           int *d_NoofNNlist, int *d_NNlist, float DL); 
                             
__global__ void UpdateNNlistWithGhostCellsPBC(int No_of_C180s, int All_Cells, float *d_CMx, float *d_CMy,float *d_CMz,
                           int Xdiv, int Ydiv, int Zdiv, float3 Subdivision_min, float3 boxMax,
                           int *d_NoofNNlist, int *d_NNlist, float DL);

__global__ void migrated_Cells_Remove_Pack(int No_of_C180s, int No_of_migration_cells_buffer, int* d_counter,
					 	int* d_migrated_cells_ind, char* d_cell_mig,
   						float *d_X,  float *d_Y,  float *d_Z,
                               		float* d_velListX, float* d_velListY, float* d_velListZ,
                               		float* d_CMx, float* d_CMy, float* d_CMz,
                               		float* d_ScaleFactor,float* d_Youngs_mod, float* d_Growth_rate, float* d_DivisionVolume,
                               		float* d_gamma_env, float* d_viscotic_damp, float* d_pressList, int* d_CellINdex, 
                               		float* d_Apo_rate, float* d_squeeze_rate,
						float *d_X_mc_buffer,  float *d_Y_mc_buffer,  float *d_Z_mc_buffer,
                               		float* d_velListX_mc_buffer, float* d_velListY_mc_buffer, float* d_velListZ_mc_buffer,
                               		float* d_CMx_mc_buffer, float* d_CMy_mc_buffer, float* d_CMz_mc_buffer,
                               		float* d_ScaleFactor_mc_buffer,float* d_Youngs_mod_mc_buffer, float* d_Growth_rate_mc_buffer, float* d_DivisionVolume_mc_buffer,
                               		float* d_gamma_env_mc_buffer, float* d_viscotic_damp_mc_buffer, float* d_pressList_mc_buffer, int* d_CellINdex_mc_buffer, 
                               		float* d_Apo_rate_mc_buffer, float* d_squeeze_rate_mc_buffer, bool colloidal_dynamics);

__global__ void migrated_Cells_Remove_Pack_PBC_X(int No_of_C180s, int No_of_migration_cells_buffer, int* d_counter,
					 	int* d_migrated_cells_ind, char* d_cell_mig, float3 boxMax,
   						float *d_X,  float *d_Y,  float *d_Z,
                               		float* d_velListX, float* d_velListY, float* d_velListZ,
                               		float* d_CMx, float* d_CMy, float* d_CMz,
                               		float* d_ScaleFactor,float* d_Youngs_mod, float* d_Growth_rate, float* d_DivisionVolume,
                               		float* d_gamma_env, float* d_viscotic_damp, float* d_pressList, int* d_CellINdex, 
                               		float* d_Apo_rate, float* d_squeeze_rate,
						float *d_X_mc_buffer,  float *d_Y_mc_buffer,  float *d_Z_mc_buffer,
                               		float* d_velListX_mc_buffer, float* d_velListY_mc_buffer, float* d_velListZ_mc_buffer,
                               		float* d_CMx_mc_buffer, float* d_CMy_mc_buffer, float* d_CMz_mc_buffer,
                               		float* d_ScaleFactor_mc_buffer,float* d_Youngs_mod_mc_buffer, float* d_Growth_rate_mc_buffer, float* d_DivisionVolume_mc_buffer,
                               		float* d_gamma_env_mc_buffer, float* d_viscotic_damp_mc_buffer, float* d_pressList_mc_buffer, int* d_CellINdex_mc_buffer, 
                               		float* d_Apo_rate_mc_buffer, float* d_squeeze_rate_mc_buffer, bool colloidal_dynamics);
                               		
__global__ void migrated_Cells_Remove_Pack_PBC_Y(int No_of_C180s, int No_of_migration_cells_buffer, int* d_counter,
					 	int* d_migrated_cells_ind, char* d_cell_mig, float3 boxMax,
   						float *d_X,  float *d_Y,  float *d_Z,
                               		float* d_velListX, float* d_velListY, float* d_velListZ,
                               		float* d_CMx, float* d_CMy, float* d_CMz,
                               		float* d_ScaleFactor,float* d_Youngs_mod, float* d_Growth_rate, float* d_DivisionVolume,
                               		float* d_gamma_env, float* d_viscotic_damp, float* d_pressList, int* d_CellINdex, 
                               		float* d_Apo_rate, float* d_squeeze_rate,
						float *d_X_mc_buffer,  float *d_Y_mc_buffer,  float *d_Z_mc_buffer,
                               		float* d_velListX_mc_buffer, float* d_velListY_mc_buffer, float* d_velListZ_mc_buffer,
                               		float* d_CMx_mc_buffer, float* d_CMy_mc_buffer, float* d_CMz_mc_buffer,
                               		float* d_ScaleFactor_mc_buffer,float* d_Youngs_mod_mc_buffer, float* d_Growth_rate_mc_buffer, float* d_DivisionVolume_mc_buffer,
                               		float* d_gamma_env_mc_buffer, float* d_viscotic_damp_mc_buffer, float* d_pressList_mc_buffer, int* d_CellINdex_mc_buffer, 
                               		float* d_Apo_rate_mc_buffer, float* d_squeeze_rate_mc_buffer, bool colloidal_dynamics);

__global__ void migrated_Cells_Remove_Pack_PBC_Z(int No_of_C180s, int No_of_migration_cells_buffer, int* d_counter,
					 	int* d_migrated_cells_ind, char* d_cell_mig, float3 boxMax,
   						float *d_X,  float *d_Y,  float *d_Z,
                               		float* d_velListX, float* d_velListY, float* d_velListZ,
                               		float* d_CMx, float* d_CMy, float* d_CMz,
                               		float* d_ScaleFactor,float* d_Youngs_mod, float* d_Growth_rate, float* d_DivisionVolume,
                               		float* d_gamma_env, float* d_viscotic_damp, float* d_pressList, int* d_CellINdex, 
                               		float* d_Apo_rate, float* d_squeeze_rate,
						float *d_X_mc_buffer,  float *d_Y_mc_buffer,  float *d_Z_mc_buffer,
                               		float* d_velListX_mc_buffer, float* d_velListY_mc_buffer, float* d_velListZ_mc_buffer,
                               		float* d_CMx_mc_buffer, float* d_CMy_mc_buffer, float* d_CMz_mc_buffer,
                               		float* d_ScaleFactor_mc_buffer,float* d_Youngs_mod_mc_buffer, float* d_Growth_rate_mc_buffer, float* d_DivisionVolume_mc_buffer,
                               		float* d_gamma_env_mc_buffer, float* d_viscotic_damp_mc_buffer, float* d_pressList_mc_buffer, int* d_CellINdex_mc_buffer, 
                               		float* d_Apo_rate_mc_buffer, float* d_squeeze_rate_mc_buffer, bool colloidal_dynamics);      

__global__ void migrated_Cells_Remove_Pack_LEbc_X(int No_of_C180s, int No_of_migration_cells_buffer, int* d_counter, float Pshift, float Vshift,
					 	int* d_migrated_cells_ind, char* d_cell_mig, float3 boxMax,
   						float *d_X,  float *d_Y,  float *d_Z,
                               		float* d_velListX, float* d_velListY, float* d_velListZ,
                               		float* d_CMx, float* d_CMy, float* d_CMz,
                               		float* d_ScaleFactor,float* d_Youngs_mod, float* d_Growth_rate, float* d_DivisionVolume,
                               		float* d_gamma_env, float* d_viscotic_damp, float* d_pressList, int* d_CellINdex, 
                               		float* d_Apo_rate, float* d_squeeze_rate,
						float *d_X_mc_buffer,  float *d_Y_mc_buffer,  float *d_Z_mc_buffer,
                               		float* d_velListX_mc_buffer, float* d_velListY_mc_buffer, float* d_velListZ_mc_buffer,
                               		float* d_CMx_mc_buffer, float* d_CMy_mc_buffer, float* d_CMz_mc_buffer,
                               		float* d_ScaleFactor_mc_buffer,float* d_Youngs_mod_mc_buffer, float* d_Growth_rate_mc_buffer, float* d_DivisionVolume_mc_buffer,
                               		float* d_gamma_env_mc_buffer, float* d_viscotic_damp_mc_buffer, float* d_pressList_mc_buffer, int* d_CellINdex_mc_buffer, 
                               		float* d_Apo_rate_mc_buffer, float* d_squeeze_rate_mc_buffer, bool colloidal_dynamics);

__global__ void migrated_cells_finder(int No_of_C180s, float *d_CM,
                         		float Sub_max, float Sub_min, float BMin, float BMax,
                         		int* d_counter_mc_r, int* d_counter_mc_l,
                         		int* d_migrated_cells_ind_R, int* d_migrated_cells_ind_L,
                         		char* d_cell_mig);

__global__ void migrated_cells_finderPBC(int No_of_C180s, float *d_CM,
                         		float Sub_max, float Sub_min, float BMin, float BMax,
                         		int* d_counter_mc_r, int* d_counter_mc_l,
                         		int* d_migrated_cells_ind_R, int* d_migrated_cells_ind_L,
                         		char* d_cell_mig);
                         		
__global__ void ghost_cells_finder_WEST(int No_of_C180s, int All_Cells, float *d_CMx, float3 Subdivision_min, 
                         		int* d_counter_gc_w, int* d_Ghost_Cells_ind_WEST);                           
                              
void SetDeviceBeforeInit();    

			
void Send_Recv_ghost_cells( int No_of_Ghost_cells_buffer, int No_of_Ghost_cells, int receiver, int sender, int tag, MPI_Comm cart_comm,
			     int shift_sender, int shift_receiver,	
			     float* X_gc_buffer, float* Y_gc_buffer, float* Z_gc_buffer, float* velListX_gc_buffer, float* velListY_gc_buffer, float* velListZ_gc_buffer,
			     float* CMx_gc_buffer, float* CMy_gc_buffer, float* CMz_gc_buffer,
			     float* X_gc, float* Y_gc, float* Z_gc, float* velListX_gc, float* velListY_gc, float* velListZ_gc, float* CMx_gc,
			     float* CMy_gc, float* CMz_gc , float* d_X, float* d_Y, float* d_Z, float* d_velListX, float* d_velListY, float* d_velListZ,
			     float* d_CMx, float* d_CMy, float* d_CMz );

void Send_Recv_migrated_cells(int No_of_migrated_cells_buffer, int No_of_migrated_cells, int receiver, int sender, int tag, MPI_Comm cart_comm, 
			     int shift_sender, int shift_receiver,
			     float* X_mc_buffer, float* Y_mc_buffer, float* Z_mc_buffer, float* velListX_mc_buffer, float* velListY_mc_buffer, float* velListZ_mc_buffer,
			     float* CMx_mc_buffer, float* CMy_mc_buffer, float* CMz_mc_buffer, float* ScaleFactor_mc_buffer, float* Youngs_mod_mc_buffer, float* Growth_rate_mc_buffer,
			     float* DivisionVolume_mc_buffer, float* gamma_env_mc_buffer, float* viscotic_damp_mc_buffer, float* pressList_mc_buffer,float* Apo_rate_mc_buffer,
			     float* squeeze_rate_mc_buffer, int* CellINdex_mc_buffer,	
			     float* X_mc, float* Y_mc, float* Z_mc, float* velListX_mc,
			     float* velListY_mc, float* velListZ_mc,float* CMx_mc, float* CMy_mc, float* CMz_mc, float* ScaleFactor_mc, float* Youngs_mod_mc, float* Growth_rate_mc,
			     float* DivisionVolume_mc, float* gamma_env_mc, float* viscotic_damp_mc, float* pressList_mc,float* Apo_rate_mc, float* squeeze_rate_mc, int* CellINdex_mc,
			     bool colloidal_dynamics);

#include<cuda.h>
#include<stdio.h>
cudaDeviceProp getDevice(void);

__global__ void  cell_division(int rank,
                               float *d_XP, float *d_YP, float *d_ZP,
                               float *d_X,  float *d_Y,  float *d_Z,
                               float* AllCMx, float* AllCMy, float* AllCMz,
                               int No_of_C180s, float *d_ran2, float repulsion_range);

__global__ void minmaxpre( int No_of_C180s, float *d_bounding_xyz,
                           float *Minx, float *Maxx, float *Miny, float *Maxy, float *Minz, float *Maxz);

__global__ void minmaxpost( int No_of_C180s,
                    float *Minx, float *Maxx, float *Miny, float *Maxy, float *Minz, float *Maxz);

__global__ void makeNNlist( int No_of_C180s, float *d_bounding_xyz,
                        float Minx, float Miny, float Minz, float attrac, int Xdiv, int Ydiv, int Zdiv,
                        int *d_NoofNNlist, int *d_NNlist, float DL);

__global__ void CenterOfMass( int No_of_C180s,
               float *d_XP, float *d_YP, float *d_ZP,
               float *CMx, float *CMy, float *CMz);

__global__ void volumes( int No_of_C180s, int *C180_56,
                         float *X,    float *Y,   float *Z,
                         float *CMx , float *CMy, float *CMz, float *vol,
                         char* cell_div, float divVol);

int printboundingbox(int rank, float *bounding_xyz);
int initialize_C180s(int Orig_No_of_C180s);
int generate_random(int no_of_ran1_vectors);
int read_fullerene_nn(void);
int read_global_params(void);
int read_json_params(const char* inpFile);
int PSSETUP(FILE* outfile);
int PSLINE(float X1, float Y1, float X2, float Y2, FILE *outfile);
int PSCIRCLE(float X,float Y,FILE *outfile);
int PSNET(int NN,int sl,float L1, float *X, float *Y, float *Z, int CCI[2][271]);

int PSNUM(float X, float Y, int NUMBER, FILE *outfile);
__global__ void propagate( int No_of_C180s, int d_C180_nn[], int d_C180_sign[],
                           float d_XP[], float d_YP[], float d_ZP[],
                           float d_X[],  float d_Y[],  float d_Z[],
                           float d_XM[], float d_YM[], float d_ZM[],
                           float *d_CMx, float *d_CMy, float *d_CMz,
                           float R0, float* d_pressList, float* d_Youngs_mod , float cellStiffness,
                           float internal_damping, float delta_t,
                           float d_bounding_xyz[],
                           float attraction_strength, float attraction_range,
                           float repulsion_strength, float repulsion_range,
                           float viscotic_damping, float mass,
                           float Minx, float Miny,  float Minz, int Xdiv, int Ydiv, int Zdiv,
                           int *d_NoofNNlist, int *d_NNlist, float DL, float gamma_visc,
                           float wall1, float wall2,
                           float threshDist, bool useWalls,
                           float* d_velListX, float* d_velListY, float* d_velListZ,
                           bool useRigidSimulationBox, float boxLength, float* d_boxMin);

__global__ void bounding_boxes( int No_of_C180s,
               float *d_XP, float *d_YP, float *d_ZP,
               float *d_X,  float *d_Y,  float *d_Z,
               float *d_XM, float *d_YM, float *d_ZM,
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

__global__ void PressureUpdate (float* d_pressList, float maxPressure,
                                float minPressure, float inc, int No_of_C180s);

__global__ void PressureReset (int* d_resetIndices, float* d_pressList,
                               float minPressure, int numResetCells);

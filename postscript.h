int getDevice(void);

__global__ void cell_division(int rank, float *XP, float *YP, float *ZP,
                                        float *X,  float *Y,  float *Z,
int No_of_C180s, float *ran2, float repulsion_range);

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
                         float *CMx , float *CMy, float *CMz, float *vol);

int printboundingbox(int rank, float *bounding_xyz);
int initialize_C180s(int Orig_No_of_C180s);
int generate_random(int no_of_ran1_vectors);
int read_fullerene_nn(void);
int read_global_params(void);
int PSSETUP(FILE *outfile);
int PSLINE(float X1, float Y1, float X2, float Y2, FILE *outfile);
int PSCIRCLE(float X,float Y,FILE *outfile);
int PSNET(int NN,int sl,float L1, float *X, float *Y, float *Z, int CCI[2][271]);

int PSNUM(float X, float Y, int NUMBER, FILE *outfile);
__global__ void propagate( int No_of_C180s, int d_C180_nn[], int C180_sign[], 
               float XP[], float YP[], float ZP[],
               float X[],  float Y[],  float Z[],
               float XM[], float YM[], float ZM[],
               float *d_CMx, float *d_CMy, float *d_CMz, 
               float R0, float Pressure, float Youngs_mod,
               float internal_damping, float delta_t,
               float bounding_xyz[],
               float attraction_strength, float attraction_range,
               float repulsion_strength, float repulsion_range,
               float viscotic_damping, float mass, 
               float Minx, float Miny,  float Minz, int Xdiv, int Ydiv, int Zdiv, 
               int *d_NoofNNlist, int *d_NNlist, float DL);

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

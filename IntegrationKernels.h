__device__ void PositionUpdate(float Vx, float Vy, float Vz,
                               float Ax, float Ay, float Az,
                               float delta_t, int index, 
                               // outputs
                               float* X, float* Y, float* Z); 

__device__ void VelocityUpdate(float Ax, float Ay, float Az,
                               float dt,
                               float VxIn, float VyIn, float VzIn,
                               // Outputs: 
                               float* Vx, float* Vy, float* Vz); 

__device__ void CalculateForce(int nodeInd, int cellInd, float nodeMass,
                               float gamma_int, float delta_t, float* pressList,
                               float Youngs_mod, float R0, float attraction_strength,
                               float attraction_range, float repulsion_strength,
                               float repulsion_range, float gamma_ext,
                               float gamma_o, float DL, 
                               float minX, float minY, float minZ,
                               int xDiv, int yDiv, float zDiv,
                               float* VxL, float* VyL, float* VzL,
                               int* d_C180_nn, int* d_C180_sign, int* d_NoofNNlist,
                               int* d_NNlist, float* d_bounding_xyz, 
                               float* d_X, float* d_Y, float* d_Z,
                               float wall1, float wall2, bool useWalls,
                               float threshDist, 
                               // Outputs: 
                               float* FxL, float* FyL, float* FzL);

__global__ void Integrate(int No_of_C180s, int* d_C180_nn, int* d_C180_sign,
                          float* d_X, float* d_Y, float* d_Z,
                          float R0, float* d_pressList, float Youngs_mod,
                          float internal_damping, float delta_t, float* d_bounding_xyz,
                          float attraction_strength, float attraction_range,
                          float repulsion_strength, float repulsion_range,
                          float viscotic_damping, float m,
                          float Minx, float Miny, float Minz,
                          float xDiv, float yDiv, float zDiv,
                          int* d_NoofNNlist, int* d_NNlist, float DL, float gamma_visc,
                          float* VxL, float* VyL, float* VzL,
                          float* FxL, float* FyL, float* FzL,
                          float wall1, float wall2, bool useWalls,
                          float threshDist);

__global__ void FirstTimeForceCalculation(int No_of_C180s, int* d_C180_nn, int* d_C180_sign,
                                          float* d_X, float* d_Y, float* d_Z,
                                          float R0, float* d_pressList, float Youngs_mod,
                                          float internal_damping, float delta_t, float* d_bounding_xyz,
                                          float attraction_strength, float attraction_range,
                                          float repulsion_strength, float repulsion_range,
                                          float viscotic_damping, float m,
                                          float Minx, float Miny, float Minz,
                                          float xDiv, float yDiv, float zDiv,
                                          int* d_NoofNNlist, int* d_NNlist, float DL, float gamma_visc,
                                          float* VxL, float* VyL, float* VzL,
                                          float* FxL, float* FyL, float* FzL,
                                          float wall1, float wall2, bool useWalls,
                                          float threshDist);

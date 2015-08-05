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
                               float* FxL, float* FyL, float* FzL){

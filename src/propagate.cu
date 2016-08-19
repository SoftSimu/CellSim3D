#include<cuda.h>
#include<vector_functions.h>
#include<stdio.h>
#include<curand_kernel.h>
#include "VectorFunctions.hpp"

__global__ void DeviceRandInit(curandState *rngStates, uint *d_seeds){
    size_t idx = threadIdx.x + blockIdx.x*blockDim.x;
    curand_init(d_seeds[idx], idx, 0, &rngStates[idx]);
}



#define check_float3(a){ \
        if (!good_float3(a)){ \
        printf("uh oh hotdog\n"); \
        printf("%d, %s, %d, %d \n", __LINE__, __FILE__, blockIdx.x, threadIdx.x);   \
        asm("trap;"); \
    } \
}

__device__ float3 GetAngleForce(const float3 iPos, const float3 kPos,
                                      const float theta_o, const float k){
    float i_dot_k = dot(iPos, kPos);
    float ri_2 = mag2(iPos);
    float rk_2 = mag2(kPos);
    //if (ri_2*rk_2 - i_dot_k*i_dot_k < 0) asm("trap;");
    float c1 = -1/( sqrtf( ri_2*rk_2 - i_dot_k*i_dot_k + 1e-3));
    
    float c2 = i_dot_k/ri_2;
    
    float theta = acos(i_dot_k/(sqrtf(ri_2)*sqrtf(rk_2) + 1e-3));
    
    float3 F_i = -k * c1 * (theta - theta_o)*(kPos - c2*iPos);
    if (!good_float3(F_i)){
        printf("c1: %f, c2: %f, theta: %f, %d %d\n", c1, c2, theta, blockIdx.x, threadIdx.x);
        printf("i.k %f ri2 %f rk2 %f, %d %d\n", i_dot_k, ri_2, rk_2, blockIdx.x, threadIdx.x);
        asm("trap;");
    }
    
    return F_i; 
}

// Watch me whip, whatch me...
__device__ void NeighNeighs (const int nodeInd, const int ni, int& nii, int& nij, int& nik,
                             const angles3 d_theta0[], float& theta1_o, float& theta2_o){
    if (nii == nodeInd){
        theta1_o = d_theta0[ni].aij;
        theta2_o = d_theta0[ni].aik; 
        nii = nij;
        nij = nik;
    } else if (nij == nodeInd){
        theta1_o = d_theta0[ni].aij;
        theta2_o = d_theta0[ni].ajk; 
        nij = nik;
    } else {
        theta1_o = d_theta0[ni].aik;
        theta2_o = d_theta0[ni].ajk; 
    }
}

// This fucking function will break if we decide to make cell geometry more interesting
__device__ float3 CalculateAngleForce(int nodeInd, int d_C180_nn[],
                                      float d_X[], float d_Y[], float d_Z[],
                                      const angles3 d_theta0[], float k, int cellInd){
    // First get the first angle contribution
    int ni = d_C180_nn[0*192 + nodeInd];
    int nj = d_C180_nn[1*192 + nodeInd];
    int nk = d_C180_nn[2*192 + nodeInd];
    
    float3 nodePos, nodeForce;

    nodeForce = make_float3(0, 0, 0);
    nodePos = make_float3(d_X[cellInd*192 + nodeInd], 
                          d_Y[cellInd*192 + nodeInd], 
                          d_Z[cellInd*192 + nodeInd]);

    float3 niPos, njPos, nkPos;
    niPos = make_float3(d_X[cellInd*192 + ni],
                        d_Y[cellInd*192 + ni],
                        d_Z[cellInd*192 + ni]);
    
    njPos = make_float3(d_X[cellInd*192 + nj],
                        d_Y[cellInd*192 + nj],
                        d_Z[cellInd*192 + nj]);

    nkPos = make_float3(d_X[cellInd*192 + nk],
                        d_Y[cellInd*192 + nk],
                        d_Z[cellInd*192 + nk]);
    
    angles3 nodeAngles = d_theta0[nodeInd]; 
    
    nodeForce = nodeForce - 
        (GetAngleForce(niPos-nodePos, njPos-nodePos, nodeAngles.aij, k) + 
         GetAngleForce(njPos-nodePos, niPos-nodePos, nodeAngles.aij, k));
    
    nodeForce = nodeForce - 
        (GetAngleForce(njPos-nodePos, nkPos-nodePos, nodeAngles.ajk, k) + 
         GetAngleForce(nkPos-nodePos, njPos-nodePos, nodeAngles.ajk, k));

    nodeForce = nodeForce -
        (GetAngleForce(niPos-nodePos, nkPos-nodePos, nodeAngles.aik, k) +
         GetAngleForce(nkPos-nodePos, niPos-nodePos, nodeAngles.aik, k));

    
    // Now second angle contributions
    // Each neighbor will have two other neighbors + our node

    int nii = d_C180_nn[ni + 0*192];
    int nij = d_C180_nn[ni + 1*192]; 
    int nik = d_C180_nn[ni + 2*192];

    int nji = d_C180_nn[nj + 0*192];
    int njj = d_C180_nn[nj + 1*192]; 
    int njk = d_C180_nn[nj + 2*192];

    int nki = d_C180_nn[nk + 0*192];
    int nkj = d_C180_nn[nk + 1*192]; 
    int nkk = d_C180_nn[nk + 2*192];
    
    float theta1_o, theta2_o; 
    float3 tiPos, tjPos;
    tiPos = make_float3(0, 0, 0);
    tjPos = make_float3(0, 0, 0);
    
    NeighNeighs(nodeInd, ni, nii, nij, nik, d_theta0, theta1_o, theta2_o);
    tiPos = make_float3(d_X[cellInd*192 + nii],
                        d_Y[cellInd*192 + nii],
                        d_Z[cellInd*192 + nii]);

    tjPos = make_float3(d_X[cellInd*192 + nij],
                        d_Y[cellInd*192 + nij],
                        d_Z[cellInd*192 + nij]);
    
    nodeForce = nodeForce + GetAngleForce(nodePos - niPos, tiPos - niPos, theta1_o, k); 
    nodeForce = nodeForce + GetAngleForce(nodePos - niPos, tjPos - niPos, theta2_o, k);
    
    NeighNeighs(nodeInd, nj, nji, njj, njk, d_theta0, theta1_o, theta2_o);
    tiPos = make_float3(d_X[cellInd*192 + nji],
                        d_Y[cellInd*192 + nji],
                        d_Z[cellInd*192 + nji]);

    tjPos = make_float3(d_X[cellInd*192 + njj],
                        d_Y[cellInd*192 + njj],
                        d_Z[cellInd*192 + njj]);

    nodeForce = nodeForce + GetAngleForce(nodePos - njPos, tiPos - njPos, theta1_o, k); 
    nodeForce = nodeForce + GetAngleForce(nodePos - njPos, tjPos - njPos, theta2_o, k);

    NeighNeighs(nodeInd, nk, nki, nkj, nkk, d_theta0, theta1_o, theta2_o);
    tiPos = make_float3(d_X[cellInd*192 + nki],
                        d_Y[cellInd*192 + nki],
                        d_Z[cellInd*192 + nki]);
                                 
    tjPos = make_float3(d_X[cellInd*192 + nkj],
                        d_Y[cellInd*192 + nkj],
                        d_Z[cellInd*192 + nkj]);

    nodeForce = nodeForce + GetAngleForce(nodePos - nkPos, tiPos - nkPos, theta1_o, k); 
    nodeForce = nodeForce + GetAngleForce(nodePos - nkPos, tjPos - nkPos, theta2_o, k);

    return nodeForce;
}
        
__global__ void CalculateForce( int No_of_C180s, int d_C180_nn[], int d_C180_sign[],
                           float d_X[],  float d_Y[],  float d_Z[],
                           float *d_CMx, float *d_CMy, float *d_CMz,
                           float R0, float* d_pressList, float* d_stiffness , float bondingYoungsMod, 
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
                                bool constrainAngles, const angles3 d_theta0[], float3* d_forceList, float r_CM_o, curandState *rngStates)
{
    // __shared__ curandState rngState;
    // if (threadIdx.x == 0){
    //     rngState = rngStates[threadIdx.x + blockDim.x*blockIdx.x];
    // }
#ifdef FORCE_DEBUG
        __shared__ float FX_sum;
        __shared__ float FY_sum;
        __shared__ float FZ_sum;
        
        if (threadIdx.x == 0){
            FX_sum = 0;
            FY_sum = 0;
            FZ_sum = 0;
        }

        __syncthreads();

#endif
        __shared__ float* dd_theta0;
        
        // for (int i =0; i<2; i++){
        //     d_theta0[i*192 + threadIdx.x] = dd_theta0[i*192 + threadIdx.x]; 
        // }

        __syncthreads(); 
        
    int rank, atom, nn_rank, nn_atom;
    int N1, N2, N3;
    int NooflocalNN;
    int localNNs[10];
    float deltaX, deltaY, deltaZ;
    float A1, A2, A3;
    float B1, B2, B3;
    float TX, TY, TZ;
    float NORM, R;
    float NX, NY, NZ;
    //float setPress;

    rank = blockIdx.x;
    atom = threadIdx.x;
    float Pressure = d_pressList[rank]; 
    int cellOffset = rank*192;
    int atomInd = cellOffset + atom;
    float stiffness;
    float delta_t = d_time[0];
    if ( rank < No_of_C180s && atom < 180 )
    {
        if (isnan(d_X[rank*192+atom]) ||
            isnan(d_Y[rank*192+atom]) || 
            isnan(d_Z[rank*192+atom])){
            printf("OH SHIT: we have a nan\n");
            printf("Particle index: %d, Cell: %d\n", atom, rank);
            printf("Crash now :(\n"); 
            asm("trap;"); 
        }

        stiffness = d_stiffness[rank];
        // printf("stiffness: %f\n", stiffness);
        // asm("trap;"); 
        
        N1 = d_C180_nn[  0+atom];
        N2 = d_C180_nn[192+atom];
        N3 = d_C180_nn[384+atom];

        A1 = d_X[rank*192+N2]-d_X[rank*192+N1];
        A2 = d_Y[rank*192+N2]-d_Y[rank*192+N1];
        A3 = d_Z[rank*192+N2]-d_Z[rank*192+N1];

        B1 = d_X[rank*192+N3]-d_X[rank*192+N1];
        B2 = d_Y[rank*192+N3]-d_Y[rank*192+N1];
        B3 = d_Z[rank*192+N3]-d_Z[rank*192+N1];

        TX = A2*B3-A3*B2;
        TY = A3*B1-A1*B3;
        TZ = A1*B2-A2*B1;

        NORM = sqrt(TX*TX+TY*TY+TZ*TZ);

        NX = d_C180_sign[atom]*TX/NORM;
        NY = d_C180_sign[atom]*TY/NORM;
        NZ = d_C180_sign[atom]*TZ/NORM;

        float X = d_X[rank*192+atom];
        float Y = d_Y[rank*192+atom];
        float Z = d_Z[rank*192+atom];

        float FX = 0.0f;
        float FY = 0.0f;
        float FZ = 0.0f;

        int nnAtomInd;
        
        
        float velX = d_velListX[atomInd];
        float velY = d_velListY[atomInd];
        float velZ = d_velListZ[atomInd];


        //  Spring Force calculation within cell
        //  go through three nearest neighbors

        float damp_const = internal_damping/delta_t;

        for ( int i = 0; i < 3 ; ++i ) // Better to open this loop
        {
            N1 = d_C180_nn[i*192+atom];

            deltaX = d_X[rank*192+N1]-d_X[rank*192+atom];
            deltaY = d_Y[rank*192+N1]-d_Y[rank*192+atom];
            deltaZ = d_Z[rank*192+N1]-d_Z[rank*192+atom];

            R  = sqrt(deltaX*deltaX+deltaY*deltaY+deltaZ*deltaZ);

            // spring forces
            FX += +stiffness*(R-R0)/R0*deltaX/R;
            FY += +stiffness*(R-R0)/R0*deltaY/R;
            FZ += +stiffness*(R-R0)/R0*deltaZ/R;

            FX += -internal_damping*(d_velListX[atomInd] - d_velListX[rank*192+N1]);
            FY += -internal_damping*(d_velListY[atomInd] - d_velListY[rank*192+N1]);
            FZ += -internal_damping*(d_velListZ[atomInd] - d_velListZ[rank*192+N1]);

        }

        // new growth force

        float3 r_CM = make_float3(d_X[atomInd] - d_CMx[rank], 
                                  d_Y[atomInd] - d_CMy[rank], 
                                  d_Z[atomInd] - d_CMz[rank]);
        float3 gForce  = make_float3(0.f, 0.f, 0.f);

        gForce = 3*Pressure*calcUnitVec(r_CM);

        //gForce = -(1e3)*(mag(r_CM) - eq_len)*calcUnitVec(r_CM); 

        FX += gForce.x; 
        FY += gForce.y; 
        FZ += gForce.z; 

        if (constrainAngles){
            float3 t = CalculateAngleForce(atom, d_C180_nn,
                                           d_X, d_Y, d_Z,
                                           d_theta0, 1000 /*Youngs_mod*/, rank);
            FX += t.x; FY += t.y; FZ += t.z;
        }
        
        
        
#ifdef FORCE_DEBUG

        atomicAdd(&FX_sum, FX);
        __syncthreads();
        atomicAdd(&FY_sum, FY);
        __syncthreads();
        atomicAdd(&FZ_sum, FZ);
        __syncthreads();
        if (threadIdx.x == 0){
            printf("Spring, pressure, internal\n");
            printf("Fx = %f, Fy = %f, Fz = %f\n", FX_sum, FY_sum, FZ_sum);
        }

#endif


        NooflocalNN = 0;

        int startx = (int)((X -Minx)/DL);
        if ( startx < 0 ) startx = 0;
        if ( startx >= Xdiv ) startx = Xdiv-1;

        int starty = (int)((Y - Miny)/DL);
        if ( starty < 0 ) starty = 0;
        if ( starty >= Ydiv ) starty = Ydiv-1;

        int startz = (int)((Z - Minz)/DL);
        if ( startz < 0 ) startz = 0;
        if ( startz >= Zdiv ) startz = Zdiv-1;

        int index = startz*Xdiv*Ydiv + starty*Xdiv + startx;

        // interfullerene attraction and repulsion
        for ( int nn_rank1 = 1 ; nn_rank1 <= d_NoofNNlist[index] ; ++nn_rank1 )
        {
            nn_rank = d_NNlist[32*index+nn_rank1-1];
            if ( nn_rank == rank ) continue;

            deltaX  = (X-d_bounding_xyz[nn_rank*6+1]>0.0f)*(X-d_bounding_xyz[nn_rank*6+1]);
            deltaX += (d_bounding_xyz[nn_rank*6+0]-X>0.0f)*(d_bounding_xyz[nn_rank*6+0]-X);

            deltaY  = (Y-d_bounding_xyz[nn_rank*6+3]>0.0f)*(Y-d_bounding_xyz[nn_rank*6+3]);
            deltaY += (d_bounding_xyz[nn_rank*6+2]-Y>0.0f)*(d_bounding_xyz[nn_rank*6+2]-Y);

            deltaZ  = (Z-d_bounding_xyz[nn_rank*6+5]>0.0f)*(Z-d_bounding_xyz[nn_rank*6+5]);
            deltaZ += (d_bounding_xyz[nn_rank*6+4]-Z>0.0f)*(d_bounding_xyz[nn_rank*6+4]-Z);

            if ( deltaX*deltaX + deltaY*deltaY + deltaZ*deltaZ > attraction_range*attraction_range )
                continue;

            ++NooflocalNN;

            //printf("NooflocalNN %d\n", NooflocalNN);

            if ( NooflocalNN > MAX_NN ){
                printf("Recoverable error: NooflocalNN = %d, should be < 8\n",NooflocalNN);
                continue;
            }

            localNNs[NooflocalNN-1] = nn_rank;
        }

        for ( int i = 0; i < NooflocalNN; ++i )
        {
            nn_rank =localNNs[i];
            nnAtomInd = nn_rank*192;

            for ( nn_atom = 0; nn_atom < 180 ; ++nn_atom )
            {
                nnAtomInd += nn_atom;

                deltaX = d_X[rank*192+atom]-d_X[nn_rank*192+nn_atom];
                deltaY = d_Y[rank*192+atom]-d_Y[nn_rank*192+nn_atom];
                deltaZ = d_Z[rank*192+atom]-d_Z[nn_rank*192+nn_atom];

                R = deltaX*deltaX+deltaY*deltaY+deltaZ*deltaZ;

                if ( R >= attraction_range*attraction_range )
                    continue;

                R = sqrt(R);

                if ( R < attraction_range )
                {
                    // FX += -attraction_strength*Youngs_mod*(attraction_range-R)/R*deltaX;
                    // FY += -attraction_strength*Youngs_mod*(attraction_range-R)/R*deltaY;
                    // FZ += -attraction_strength*Youngs_mod*(attraction_range-R)/R*deltaZ;

                    FX += -500*(attraction_range-R)/R*deltaX;
                    FY += -500*(attraction_range-R)/R*deltaY;
                    FZ += -500*(attraction_range-R)/R*deltaZ;

                    // hinder rearrangements

                    // First calculate relative velocity
                    float v_ijx = d_velListX[atomInd] - d_velListX[nn_rank*192+nn_atom];
                    float v_ijy = d_velListY[atomInd] - d_velListY[nn_rank*192+nn_atom];
                    float v_ijz = d_velListZ[atomInd] - d_velListZ[nn_rank*192+nn_atom];

                    // Dot product between relative and normal to surface
                    float vijDotn = v_ijx*NX + v_ijy*NY + v_ijz*NZ; 

                    // Tangential component
                    float vTauX = v_ijx - vijDotn*NX;
                    float vTauY = v_ijy - vijDotn*NY;
                    float vTauZ = v_ijz - vijDotn*NZ; 

                    FX -= viscotic_damping*vTauX;
                    FY -= viscotic_damping*vTauY;
                    FZ -= viscotic_damping*vTauZ;
                }
                if ( R < repulsion_range )
                {
                    if (R < (repulsion_range-0.01)) R = repulsion_range-0.01; 
                    // FX += +repulsion_strength*Youngs_mod*(repulsion_range-R)/R*deltaX;
                    // FY += +repulsion_strength*Youngs_mod*(repulsion_range-R)/R*deltaY;
                    // FZ += +repulsion_strength*Youngs_mod*(repulsion_range-R)/R*deltaZ;

                    FX += +100000*(repulsion_range-R)/R*deltaX;
                    FY += +100000*(repulsion_range-R)/R*deltaY;
                    FZ += +100000*(repulsion_range-R)/R*deltaZ;
                    // if ( deltaX*(d_CMx[rank]-d_CMx[nn_rank])  +
                    //      deltaY*(d_CMy[rank]-d_CMy[nn_rank])  +
                    //      deltaZ*(d_CMz[rank]-d_CMz[nn_rank]) < 0.0f )
                    // {
                    //     //printf("fullerene %d inside %d?\n",rank, nn_rank);
                    // }
                }

            }

        }

#ifdef FORCE_DEBUG

        if (threadIdx.x == 0){
            FX_sum = 0;
            FY_sum = 0;
            FZ_sum = 0;
        }
        __syncthreads();

        atomicAdd(&FX_sum, FX);
        __syncthreads();
        atomicAdd(&FY_sum, FY);
        __syncthreads();
        atomicAdd(&FZ_sum, FZ);
        __syncthreads();
        if (threadIdx.x == 0){
            printf("neighbours\n");
            printf("Fx = %f, Fy = %f, Fz = %f\n", FX_sum, FY_sum, FZ_sum);
        }



#endif

        // add friction

        FX += -1 * gamma_visc * velX;
        FY += -1 * gamma_visc * velY;
        FZ += -1 * gamma_visc * velZ;

        if (useWalls == 1){
            // Wall repulsion

            float wallDist = d_Z[rank*192+atom] - wall1;
            
            if (abs(wallDist) <= threshDist){
            //repulse
                if (wallDist * FZ < 0) FZ = -FZ;
            }

            wallDist = d_Z[rank*192+atom] - wall2;
            if (abs(wallDist) <= threshDist){
                
                if (wallDist * FZ < 0) FZ = -FZ;

            }

            __syncthreads();
        }

        // add forces from simulation box if needed:

        if (useRigidSimulationBox){
            float gap1, gap2;
            
            // X
            
            gap1 = d_X[atomInd] - d_boxMin[0];
            gap2 = d_boxMin[0] + boxLength - d_X[atomInd];
            
            if (abs(gap1) < threshDist && gap1*FX < 0) FX = -FX;
            if (abs(gap2) < threshDist && gap2*FX < 0) FX = -FX;

            // Y

            gap1 = d_Y[atomInd] - d_boxMin[1];
            gap2 = d_boxMin[1] + boxLength - d_Y[atomInd];

            if (abs(gap1) < threshDist && gap1*FY < 0) FY = -FY;
            if (abs(gap2) < threshDist && gap2*FY < 0) FY = -FY;

            // Z
            gap1 = d_Z[atomInd] - d_boxMin[2];
            gap2 = d_boxMin[2] + boxLength - d_Z[atomInd];

            if (abs(gap1) < threshDist && gap1*FZ < 0) FZ = -FZ;
            if (abs(gap2) < threshDist && gap2*FZ < 0) FZ = -FZ;
        }

        d_forceList[rank*192+atom] = make_float3(FX, FY, FZ); 
    }
}


__global__ void Integrate(float *d_XP, float *d_YP, float *d_ZP,
                          float *d_X, float *d_Y, float *d_Z, 
                          float *d_XM, float *d_YM, float *d_ZM,
                          float *d_velListX, float *d_velListY, float *d_velListZ, 
                          float *d_time, float mass,
                          float3 *d_forceList, int numCells, curandState *rngStates, float3 *d_rands){
    const int cellInd = blockIdx.x;
    const int node = threadIdx.x;
    const float delta_t = d_time[0];
    
    
    if (cellInd < numCells && node < 180){
        int nodeInd = cellInd*192 + node; 
        float FX = d_forceList[nodeInd].x;
        float FY = d_forceList[nodeInd].y;
        float FZ = d_forceList[nodeInd].z;
        curandState rngState = rngStates[nodeInd];
        
        float3 r = 0.001*make_float3(curand_uniform(&rngState) - 0.5f,
                                     curand_uniform(&rngState) - 0.5f,
                                     curand_uniform(&rngState) - 0.5f);

        d_rands[nodeInd] = r; 
        
        d_XP[nodeInd] =
            1.0/(1.0+delta_t/(2*mass))*
            ((delta_t*delta_t/mass)*FX+2*d_X[nodeInd]+(delta_t/(2*mass)-1.0)*d_XM[nodeInd]) + r.x;
        
        d_YP[nodeInd] =
            1.0/(1.0+delta_t/(2*mass))*
            ((delta_t*delta_t/mass)*FY+2*d_Y[nodeInd]+(delta_t/(2*mass)-1.0)*d_YM[nodeInd]) + r.y;
        
        d_ZP[nodeInd] =
            1.0/(1.0+delta_t/(2*mass))*
            ((delta_t*delta_t/mass)*FZ+2*d_Z[nodeInd]+(delta_t/(2*mass)-1.0)*d_ZM[nodeInd]) + r.z;
        
        d_velListX[nodeInd] = (d_XP[nodeInd] - d_XM[nodeInd])/(2*delta_t); 
        d_velListY[nodeInd] = (d_YP[nodeInd] - d_YM[nodeInd])/(2*delta_t); 
        d_velListZ[nodeInd] = (d_ZP[nodeInd] - d_ZM[nodeInd])/(2*delta_t);

        rngStates[nodeInd] = rngState;
    }
}

__global__ void ForwardTime(float *d_XP, float *d_YP, float *d_ZP,
                            float *d_X, float *d_Y, float *d_Z,
                            float *d_XM, float *d_YM, float *d_ZM, 
                            int numCells){
    
    const int nodeInd = blockIdx.x*blockDim.x + threadIdx.x;
    if (nodeInd < 192*numCells){

        d_XM[nodeInd] = d_X[nodeInd]; 
        d_YM[nodeInd] = d_Y[nodeInd]; 
        d_ZM[nodeInd] = d_Z[nodeInd];
    
        d_X[nodeInd] = d_XP[nodeInd];
        d_Y[nodeInd] = d_YP[nodeInd];
        d_Z[nodeInd] = d_ZP[nodeInd];
    }
}

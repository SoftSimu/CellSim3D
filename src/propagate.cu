#include<cuda.h>
#include<stdio.h>
#include "VectorFunctions.hpp"

__device__ float3 GetAngleForce(const float3 iPos, const float3 kPos,
                                      const float theta_o, const float k){
    float i_dot_k = dot(iPos, kPos);
    float ri_2 = mag2(iPos);
    float rk_2 = mag2(kPos);
    
    float c1 = 1/( sqrtf( ri_2*rk_2 - i_dot_k*i_dot_k ) );
    
    float c2 = i_dot_k/ri_2;
        
    float theta = acosf(i_dot_k/(sqrtf(ri_2)*sqrtf(rk_2)));

    float3 F_i = -k * c1 * (theta - theta_o)*(kPos - c2*iPos);
    
    
    return F_i; 
}


__device__ void NeighNeighs (int& nodeInd, int& n11, int& n12, int& n13){
    if (n11 == nodeInd){
        n11 = n12;
        n12 = n13;
    } else if (n12 == nodeInd){
        n12 = n13;
    }
}

// This fucking fuction will break if we decide to make cell geometry more interesting
__device__ float3 CalculateAngleForce(int nodeInd, int d_C180_nn[],
                                      float d_X[], float d_Y[], float d_Z[],
                                      float theta_o, float k, int cellInd){
    // First get the first angle contribution
    int n1 = d_C180_nn[0*192 + nodeInd];
    int n2 = d_C180_nn[1*192 + nodeInd];
    int n3 = d_C180_nn[2*192 + nodeInd];
    
    float3 nodePos, nodeForce;

    nodeForce.x = 0; nodeForce.y = 0; nodeForce.z=0; 

    nodePos.x = d_X[cellInd*192 + nodeInd];
    nodePos.y = d_Y[cellInd*192 + nodeInd];
    nodePos.z = d_Z[cellInd*192 + nodeInd];

    float3 n1Pos, n2Pos, n3Pos;
    
    n1Pos.x = d_X[cellInd*192 + n1];
    n1Pos.y = d_Y[cellInd*192 + n1];
    n1Pos.z = d_Z[cellInd*192 + n1];
    
    n2Pos.x = d_X[cellInd*192 + n2];
    n2Pos.y = d_Y[cellInd*192 + n2];
    n2Pos.z = d_Z[cellInd*192 + n2];
    
    n3Pos.x = d_X[cellInd*192 + n3];
    n3Pos.y = d_Y[cellInd*192 + n3];
    n3Pos.z = d_Z[cellInd*192 + n3];
    

    // i = n1, k = n2
    nodeForce = nodeForce - (GetAngleForce(n1Pos-nodePos, n2Pos-nodePos, theta_o, k) + GetAngleForce(n2Pos-nodePos, n1Pos-nodePos, theta_o, k));
    // i = n1, k = n3
    nodeForce = nodeForce - (GetAngleForce(n1Pos-nodePos, n3Pos-nodePos, theta_o, k) + GetAngleForce(n3Pos-nodePos, n3Pos-nodePos, theta_o, k));
    // i = n2, k = n3
    nodeForce = nodeForce - (GetAngleForce(n2Pos-nodePos, n3Pos-nodePos, theta_o, k) + GetAngleForce(n3Pos-nodePos, n2Pos-nodePos, theta_o, k));
    
    // Now second angle contributions
    // Each neighbor will have two other neighbors + our node

    int n11 = d_C180_nn[n1 + 0*192];
    int n12 = d_C180_nn[n1 + 1*192]; 
    int n13 = d_C180_nn[n1 + 2*192];

    int n21 = d_C180_nn[n2 + 0*192];
    int n22 = d_C180_nn[n2 + 1*192]; 
    int n23 = d_C180_nn[n2 + 2*192];

    int n31 = d_C180_nn[n3 + 0*192];
    int n32 = d_C180_nn[n3 + 1*192]; 
    int n33 = d_C180_nn[n3 + 2*192];

    // Watch me whip... Watch me
    NeighNeighs(nodeInd, n11, n12, n13);
    NeighNeighs(nodeInd, n21, n22, n23);
    NeighNeighs(nodeInd, n31, n32, n33); 

    // n11, n12 are the neighbours of n1 that are not our node
    // Now we must calculate the force on nodeInd due to it being part of the angle with n11, n12, and n1
    // n1 will be defined as the origin
    float3 t1Pos, t2Pos;
    
    t1Pos.x = d_X[cellInd*192 + n11]; 
    t1Pos.y = d_Y[cellInd*192 + n11]; 
    t1Pos.z = d_Z[cellInd*192 + n11];

    t2Pos.x = d_X[cellInd*192 + n12]; 
    t2Pos.y = d_Y[cellInd*192 + n12]; 
    t2Pos.z = d_Z[cellInd*192 + n12];
    
    nodeForce = nodeForce + GetAngleForce(nodePos - n1Pos, t1Pos - n1Pos, theta_o, k); 
    nodeForce = nodeForce + GetAngleForce(nodePos - n1Pos, t2Pos - n1Pos, theta_o, k);
    
    // n21, n22
    t1Pos.x = d_X[cellInd*192 + n21]; 
    t1Pos.y = d_Y[cellInd*192 + n21]; 
    t1Pos.z = d_Z[cellInd*192 + n21];

    t2Pos.x = d_X[cellInd*192 + n22]; 
    t2Pos.y = d_Y[cellInd*192 + n22]; 
    t2Pos.z = d_Z[cellInd*192 + n22];
    
    nodeForce = nodeForce + GetAngleForce(nodePos - n2Pos, t1Pos - n2Pos, theta_o, k); 
    nodeForce = nodeForce + GetAngleForce(nodePos - n2Pos, t2Pos - n2Pos, theta_o, k);

    // n31, n32
    t1Pos.x = d_X[cellInd*192 + n31]; 
    t1Pos.y = d_Y[cellInd*192 + n31]; 
    t1Pos.z = d_Z[cellInd*192 + n31];
                                 
    t2Pos.x = d_X[cellInd*192 + n32]; 
    t2Pos.y = d_Y[cellInd*192 + n32]; 
    t2Pos.z = d_Z[cellInd*192 + n32];

    nodeForce = nodeForce + GetAngleForce(nodePos - n2Pos, t1Pos - n3Pos, theta_o, k); 
    nodeForce = nodeForce + GetAngleForce(nodePos - n2Pos, t2Pos - n3Pos, theta_o, k);

    return nodeForce;
}
        
__global__ void propagate( int No_of_C180s, int d_C180_nn[], int d_C180_sign[],
                           float d_XP[], float d_YP[], float d_ZP[],
                           float d_X[],  float d_Y[],  float d_Z[],
                           float d_XM[], float d_YM[], float d_ZM[],
                           float *d_CMx, float *d_CMy, float *d_CMz,
                           float R0, float* d_pressList, float* d_stiffness , float bondingYoungsMod, 
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
                           bool useRigidSimulationBox, float boxLength, float* d_boxMin, float Youngs_mod)
{
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
    
    if ( rank < No_of_C180s && atom < 180 )
    {
        if (isnan(d_X[rank*192+atom]) ||
            isnan(d_Y[rank*192+atom]) || 
            isnan(d_Z[rank*192+atom])){
            printf("OH SHIT: we have a nan\n");
            printf("Particle index: %d\n", atom);
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


            // pressure forces
            FX += Pressure*NX;
            FY += Pressure*NY;
            FZ += Pressure*NZ;

            // internal damping
            // FX += -damp_const*(-deltaX-(d_XM[rank*192+atom]-d_XM[rank*192+N1]));
            // FY += -damp_const*(-deltaY-(d_YM[rank*192+atom]-d_YM[rank*192+N1]));
            // FZ += -damp_const*(-deltaZ-(d_ZM[rank*192+atom]-d_ZM[rank*192+N1]));

            FX += -internal_damping*(d_velListX[atomInd] - d_velListX[rank*192+N1]);
            FY += -internal_damping*(d_velListY[atomInd] - d_velListY[rank*192+N1]);
            FZ += -internal_damping*(d_velListZ[atomInd] - d_velListZ[rank*192+N1]);

        }

        
        float3 t = CalculateAngleForce(atom, d_C180_nn,
                                       d_X, d_Y, d_Z,
                                       3.14159*0.666666, 1000, rank);
            
        printf("Angle Force on node %d= (%f, %f, %f)\n", atom, t.x, t.y, t.z); 
        asm("trap;"); 
        
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

            if ( NooflocalNN > 10 ){
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
                    FX += -attraction_strength*Youngs_mod*(attraction_range-R)/R*deltaX;
                    FY += -attraction_strength*Youngs_mod*(attraction_range-R)/R*deltaY;
                    FZ += -attraction_strength*Youngs_mod*(attraction_range-R)/R*deltaZ;

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
                    FX += +repulsion_strength*Youngs_mod*(repulsion_range-R)/R*deltaX;
                    FY += +repulsion_strength*Youngs_mod*(repulsion_range-R)/R*deltaY;
                    FZ += +repulsion_strength*Youngs_mod*(repulsion_range-R)/R*deltaZ;

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

        // time propagation
        
        d_XP[rank*192+atom] =
            1.0/(1.0+delta_t/(2*mass))*
            ((delta_t*delta_t/mass)*FX+2*d_X[rank*192+atom]+(delta_t/(2*mass)-1.0)*d_XM[rank*192+atom]);
        d_YP[rank*192+atom] =
            1.0/(1.0+delta_t/(2*mass))*
            ((delta_t*delta_t/mass)*FY+2*d_Y[rank*192+atom]+(delta_t/(2*mass)-1.0)*d_YM[rank*192+atom]);
        d_ZP[rank*192+atom] =
            1.0/(1.0+delta_t/(2*mass))*
            ((delta_t*delta_t/mass)*FZ+2*d_Z[rank*192+atom]+(delta_t/(2*mass)-1.0)*d_ZM[rank*192+atom]);

        d_velListX[rank*192+atom] = (d_XP[atomInd] - d_XM[atomInd])/(2*delta_t); 
        d_velListY[rank*192+atom] = (d_YP[atomInd] - d_YM[atomInd])/(2*delta_t); 
        d_velListZ[rank*192+atom] = (d_ZP[atomInd] - d_ZM[atomInd])/(2*delta_t);

    }
}

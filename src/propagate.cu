#include<cuda.h>
#include<vector_functions.h>
#include<stdio.h>
#include<curand_kernel.h>
#include "VectorFunctions.hpp"
#include "globals.cuh"
#include "State.cuh"

__global__ void DeviceRandInit(curandState *rngStates, uint *d_seeds, unsigned long long num){
    size_t idx = threadIdx.x + blockIdx.x*blockDim.x;
    if (idx < num){
        curandState rS = rngStates[idx];
        curand_init(d_seeds[idx], 0, 0 , &rS);
        rngStates[idx] = rS; 
    }
}



#define check_float3(a){ \
        if (!good_float3(a)){ \
        printf("uh oh hotdog\n"); \
        printf("%d, %s, %d, %d \n", __LINE__, __FILE__, blockIdx.x, threadIdx.x);   \
        asm("trap;"); \
    } \
}

__device__ real3 GetAngleForce(const real3 iPos, const real3 kPos,
                                      const real theta_o, const real k){
    real i_dot_k = dot(iPos, kPos);
    real ri_2 = mag2(iPos);
    real rk_2 = mag2(kPos);
    //if (ri_2*rk_2 - i_dot_k*i_dot_k < 0) asm("trap;");
    real c1 = -1/( sqrtf( ri_2*rk_2 - i_dot_k*i_dot_k + 1e-3));
    
    real c2 = i_dot_k/ri_2;
    
    real theta = acos(i_dot_k/(sqrtf(ri_2)*sqrtf(rk_2) + 1e-3));
    
    real3 F_i = -k * c1 * (theta - theta_o)*(kPos - c2*iPos);

    // real imag = mag(iPos);
    // real kmag = mag(kPos);
    
    // real cos_theta = dot(iPos, kPos)/(imag*kmag);
    // real cos_thetao = cos(theta_o); 

    // real3 F_i =  -2*k/(imag*kmag) * (cos_theta - cos_thetao) * kPos; 
    
    if (!good_real3(F_i)){
        printf("c1: %f, c2: %f, theta: %f, %d %d\n", c1, c2, theta, blockIdx.x, threadIdx.x);
        printf("i.k %f ri2 %f rk2 %f, %d %d\n", i_dot_k, ri_2, rk_2, blockIdx.x, threadIdx.x);
        asm("trap;");
    }

    // if (!good_real3(F_i)){
    //     printf("Angle force calculation failed for node %d in cell %d\n", blockIdx.x, threadIdx.x);
    //     asm("trap;");
    // }
    
    return F_i; 
}

// Watch me whip, whatch me...
__device__ void NeighNeighs (const int nodeInd, const int ni, int& nii, int& nij, int& nik,
                             const angles3 d_theta0[], real& theta1_o, real& theta2_o){
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
__device__ real3 CalculateAngleForce(int nodeInd, int d_C180_nn[],
                                      real d_X[], real d_Y[], real d_Z[],
                                      const angles3 d_theta0[], real k, int cellInd){
    // First get the first angle contribution
    int ni = d_C180_nn[0*192 + nodeInd];
    int nj = d_C180_nn[1*192 + nodeInd];
    int nk = d_C180_nn[2*192 + nodeInd];
    
    real3 nodePos, nodeForce;

    nodeForce = make_real3(0, 0, 0);
    nodePos = make_real3(d_X[cellInd*192 + nodeInd], 
                         d_Y[cellInd*192 + nodeInd], 
                         d_Z[cellInd*192 + nodeInd]);

    real3 niPos, njPos, nkPos;
    niPos = make_real3(d_X[cellInd*192 + ni],
                       d_Y[cellInd*192 + ni],
                       d_Z[cellInd*192 + ni]);
    
    njPos = make_real3(d_X[cellInd*192 + nj],
                       d_Y[cellInd*192 + nj],
                       d_Z[cellInd*192 + nj]);

    nkPos = make_real3(d_X[cellInd*192 + nk],
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
    
    real theta1_o, theta2_o; 
    real3 tiPos, tjPos;
    tiPos = make_real3(0, 0, 0);
    tjPos = make_real3(0, 0, 0);
    
    NeighNeighs(nodeInd, ni, nii, nij, nik, d_theta0, theta1_o, theta2_o);
    tiPos = make_real3(d_X[cellInd*192 + nii],
                        d_Y[cellInd*192 + nii],
                        d_Z[cellInd*192 + nii]);

    tjPos = make_real3(d_X[cellInd*192 + nij],
                        d_Y[cellInd*192 + nij],
                        d_Z[cellInd*192 + nij]);
    
    nodeForce = nodeForce + GetAngleForce(nodePos - niPos, tiPos - niPos, theta1_o, k); 
    nodeForce = nodeForce + GetAngleForce(nodePos - niPos, tjPos - niPos, theta2_o, k);
    
    NeighNeighs(nodeInd, nj, nji, njj, njk, d_theta0, theta1_o, theta2_o);
    tiPos = make_real3(d_X[cellInd*192 + nji],
                        d_Y[cellInd*192 + nji],
                        d_Z[cellInd*192 + nji]);

    tjPos = make_real3(d_X[cellInd*192 + njj],
                        d_Y[cellInd*192 + njj],
                        d_Z[cellInd*192 + njj]);

    nodeForce = nodeForce + GetAngleForce(nodePos - njPos, tiPos - njPos, theta1_o, k); 
    nodeForce = nodeForce + GetAngleForce(nodePos - njPos, tjPos - njPos, theta2_o, k);

    NeighNeighs(nodeInd, nk, nki, nkj, nkk, d_theta0, theta1_o, theta2_o);
    tiPos = make_real3(d_X[cellInd*192 + nki],
                        d_Y[cellInd*192 + nki],
                        d_Z[cellInd*192 + nki]);
                                 
    tjPos = make_real3(d_X[cellInd*192 + nkj],
                        d_Y[cellInd*192 + nkj],
                        d_Z[cellInd*192 + nkj]);

    nodeForce = nodeForce + GetAngleForce(nodePos - nkPos, tiPos - nkPos, theta1_o, k); 
    nodeForce = nodeForce + GetAngleForce(nodePos - nkPos, tjPos - nkPos, theta2_o, k);

    return nodeForce;
}
        
__global__ void CalculateConForce(SimStatePtrs simState, sim_params_struct sim_params,
                                  int Xdiv, int Ydiv, int Zdiv){

    __shared__ int no_of_cells;
    __shared__ int num_nodes;
    __shared__ real* pressList;
    __shared__ real* d_X; 
    __shared__ real* d_Y; 
    __shared__ real* d_Z;
    __shared__ int* C180_nn;
    __shared__ int* C180_sign;
    __shared__ real* R0;
    __shared__ real3 cellCOM;
    __shared__ real Pressure;
    __shared__ real stiffness;
    __shared__ real* d_bounding_xyz;
    __shared__ real attr_range;
    __shared__ real attr_stren;
    __shared__ real rep_range;
    __shared__ real attr_stren;
    __shared__ real thresh_dist;
    __shared__ real3 box_max;
    __shared__ real* d_Fx; 
    __shared__ real* d_Fy; 
    __shared__ real* d_Fz;
    __shared__ real3* d_theta0;
    
    size_t cellInd, localNodeInd;
    
    cellInd = blockIdx.x;
    localNodeInd = threadIdx.x;
    
    size_t cellOffset = rank*192;
    size_t nodeInd = cellOffset + localNodeInd;
    if ( cellInd < simState.no_of_cells && localNodeInd < 180 )
    {
        if (threadIdx.x == 0){
            no_of_cells = simState.no_of_cells;
            num_nodes = 192;
            d_X = simState.pos.x; 
            d_Y = simState.pos.y; 
            d_Z = simState.pos.z;
            C180_nn = simState.C180_nn;
            C180_sign = simState.C180_sign;
            R0 = simState.R0;
            d_CMx = simState.cellCOMs.x;
            d_CMx = simState.cellCOMs.y;
            d_CMx = simState.cellCOMs.z;
            Pressure = simState.pressures[rank];
            stiffness = simState.youngsMod[rank];
            
            cellCOM = make_real3(simState.cellCOMs.x[rank],
                                 simState.cellCOMs.y[rank],
                                 simState.cellCOMS.z[rank]);
            
            attr_range = sim_params.core_params.attr_range;
            attr_stren = sim_params.core_params.attr_stiff;
            
            rep_range = sim_params.core_params.rep_range;
            rep_stren = sim_params.core_params.rep_stiff;
            thresh_dist = sim_params.box_params.thresh_dist;
            box_max = sim_params.box_params.box_len;
            d_Fx = simState.conForce.x; 
            d_Fy = simState.conForce.y; 
            d_Fz = simState.conForce.z;
            d_theta0 = simState.theta0;
        }
        __syncthreads();
        
        real3 nodePos = make_real3(d_X[nodeInd],
                                   d_Y[nodeInd],
                                   d_Z[nodeInd]);
        real3 force = make_real3(0, 0, 0); 
        
        if (!good_real3(nodePos)){
            printf("OH SHIT: we have a nan\n");
            printf("Particle index: %d, Cell: %d\n", atom, rank);
            printf("Crash now :(\n"); 
            asm("trap;"); 
        }

        //  Spring Force calculation within cell
        //  go through three nearest neighbors
        for ( int i = 0; i < 3 ; ++i ) // Maybe Better to open this loop
        {
            int N1 = d_C180_nn[i*192+atom];

            real3 delta = make_real3(d_X[cellInd*192+N1],
                                     d_Y[cellInd*192+N1],
                                     d_Z[cellInd*192+N1]) - nodePos;

            real R0 = d_R0[i*192 + atom];

            //spring forces
            force = force + (stiffness*(R-R0)/R0)*calcUnitVec(delta);
        }

        // new growth force
        real3 rCM = nodePos - cellCOM;
        
        force = force + 3*Pressure*calcUnitVec(rCM); //why the 3? Don't know:)

        //gForce = -10*(volList[cellInd] - div_vol)*calcUnitVec(r_CM);
        //gForce = -10*(mag(r_CM) - r_CM_o)*calcUnitVec(r_CM);
        
        if (constrainAngles){
            real3 t = CalculateAngleForce(localNodeInd, d_C180_nn,
                                          d_X, d_Y, d_Z,
                                          d_theta0, stiffness, cellInd);
            force = force + t;
        }
        
        
        // interfullerene attraction and repulsion
        
        NooflocalNN = 0;

        int startx = (int)((nodePos.x - sim_state.mins.x[0])/DL);
        if ( startx < 0 ) startx = 0;
        if ( startx >= Xdiv ) startx = Xdiv-1;

        int starty = (int)((nodePos.y - sim_state.mins.y[0])/DL);
        if ( starty < 0 ) starty = 0;
        if ( starty >= Ydiv ) starty = Ydiv-1;

        int startz = (int)((nodePos.z - sim_state.mins.z[0])/DL);
        if ( startz < 0 ) startz = 0;
        if ( startz >= Zdiv ) startz = Zdiv-1;

        int index = startz*Xdiv*Ydiv + starty*Xdiv + startx;
        

        int nn_cellInd;
        real deltaX, deltaY, deltaZ;
        int localNNs[MAX_NN];
        for ( int nn_rank1 = 1 ; nn_rank1 <= d_NoofNNlist[index] ; ++nn_rank1 ){
            nn_cellInd = d_NNlist[32*index+nn_rank1-1];
            
            if ( nn_rank == rank )
                continue;

            deltaX  = (pos.x-d_bounding_xyz[nn_rank*6+1]>0.0f)*(pos.x-d_bounding_xyz[nn_rank*6+1]);
            deltaX += (d_bounding_xyz[nn_rank*6+0]-pos.x>0.0f)*(d_bounding_xyz[nn_rank*6+0]-pos.x);

            deltaY  = (pos.y-d_bounding_xyz[nn_rank*6+3]>0.0f)*(pos.y-d_bounding_xyz[nn_rank*6+3]);
            deltaY += (d_bounding_xyz[nn_rank*6+2]-pos.y>0.0f)*(d_bounding_xyz[nn_rank*6+2]-pos.y);

            deltaZ  = (pos.z-d_bounding_xyz[nn_rank*6+5]>0.0f)*(pos.z-d_bounding_xyz[nn_rank*6+5]);
            deltaZ += (d_bounding_xyz[nn_rank*6+4]-pos.z>0.0f)*(d_bounding_xyz[nn_rank*6+4]-pos.z);

            if (deltaX*deltaX + deltaY*deltaY + deltaZ*deltaZ >
                attr_range*attr_range)
                continue;

            ++NooflocalNN;

            //printf("NooflocalNN %d\n", NooflocalNN);

            if ( NooflocalNN > MAX_NN ){
                printf("Recoverable error: NooflocalNN = %d, should be < 8\n", NooflocalNN);
                continue;
            }

            localNNs[NooflocalNN-1] = nn_rank;
        }

        real3 contactForce = make_real3(0.f, 0.f, 0.f);
        for ( int i = 0; i < NooflocalNN; ++i ){
            int nn_cellInd = localNNs[i];
            int nnNodeInd = nn_cellInd*192;

            for ( int nn_node = 0; nn_node < 180 ; ++nn_node){
                nnNodeInd += nn_node;
                
                real3 Rvec = nodePos - make_real3(d_X[nnNodeInd],
                                                  d_Y[nnNodeInd],
                                                  d_Z[nnNodeInd]);

                real R = mag2(Rvec)
                if (R > attr_range*attr_range)
                    continue;

                R = sqrtf(R);
                
                contactForce = -attr_stren*youngs_mod*(attr_range-R)*calcUnitVec(Rvec);
                
                if (R <= rep_range){
                    //if (R < (rep_range-0.01)) R = repulsion_range-0.01;
                                        
                    contactForce = contactForce +
                        rep_stren*stiffness*(rep_range-R)*calcUnitVec(Rvec);
                }
            }

        }

        // add forces from simulation box if needed:
        if (useRigidSimulationBox){
            real gap1, gap2; 

            gap1 = nodePos.x; 
            gap2 = box_max.x - nodePos.x; 

            if (gap1 < thresh_dist){
                force.x += -rep_stren*stiffness*(gap1-thresh_dist);
            }

            if (gap2 < threshDist){
                force.x += rep_stren*stiffness*(gap2 - thresh_dist);
            }

            gap1 = nodePos.y; 
            gap2 = box_max.y - nodePos.y; 

            if (gap1 < thresh_dist){
                force.y += -rep_stren*stiffness*(gap1-thresh_dist);
            }

            if (gap2 < threshDist){
                force.y += rep_stren*stiffness*(gap2 - thresh_dist);
            }
            
            gap1 = nodePos.z; 
            gap2 = box_max.z - nodePos.z; 

            if (gap1 < thresh_dist){
                force.z += -rep_stren*stiffness*(gap1-thresh_dist);
            }

            if (gap2 < threshDist){
                force.z += rep_stren*stiffness*(gap2 - thresh_dist);
            }
            
        }

        d_Fx[nodeInd] = force.x;
        d_Fy[nodeInd] = force.y;
        d_Fz[nodeInd] = force.z;
    }
}


__global__ void CalculateDisForce( int No_of_C180s, int d_C180_nn[], int d_C180_sign[],
                                   real d_X[],  real d_Y[],  real d_Z[],
                                   real gamma_int,
                                   real d_bounding_xyz[],
                                   real attraction_range,
                                   real gamma_ext,
                                   real Minx, real Miny,  real Minz, int Xdiv, int Ydiv, int Zdiv,
                                   int *d_NoofNNlist, int *d_NNlist, real DL, real gamma_o,
                                   real* d_velListX, real* d_velListY, real* d_velListZ,
                                   R3Nptrs d_fDisList){
    size_t cellInd = blockIdx.x;
    size_t nodeInd = threadIdx.x;

    if (cellInd < No_of_C180s && nodeInd < 180){
        size_t globalNodeInd = cellInd*192 + nodeInd;
        size_t N = 0;
        real3 force = make_real3(0, 0, 0);
        real3 nodeVelocity = make_real3(d_velListX[globalNodeInd],
                                          d_velListY[globalNodeInd],
                                          d_velListZ[globalNodeInd]);

        real3 neighVelocity = make_real3(0, 0, 0);
        
        // dampen bonding
        for (int i = 0; i < 3; ++i){
            N = d_C180_nn[i*192+nodeInd];
            neighVelocity = make_real3(d_velListX[cellInd*192+N],
                                        d_velListY[cellInd*192+N],
                                        d_velListZ[cellInd*192+N]);
                
            force = force - gamma_int*(nodeVelocity - neighVelocity);
        }

        // inter-cellular friction
        // This break if cell geometry every changes
        int N1 = d_C180_nn[  0+nodeInd];
        int N2 = d_C180_nn[192+nodeInd];
        int N3 = d_C180_nn[384+nodeInd];

        real3 A = make_real3(d_X[cellInd*192+N2]-d_X[cellInd*192+N1],
                               d_Y[cellInd*192+N2]-d_Y[cellInd*192+N1],
                               d_Z[cellInd*192+N2]-d_Z[cellInd*192+N1]);

        real3 B = make_real3(d_X[cellInd*192+N3]-d_X[cellInd*192+N1],
                               d_Y[cellInd*192+N3]-d_Y[cellInd*192+N1],
                               d_Z[cellInd*192+N3]-d_Z[cellInd*192+N1]);

        real3 normal = calcUnitVec(cross(A, B));

        normal = d_C180_sign[nodeInd]*normal;

        real X = d_X[globalNodeInd];
        real Y = d_Y[globalNodeInd];
        real Z = d_Z[globalNodeInd];

        real deltaX = 0;
        real deltaY = 0;
        real deltaZ = 0;
        real R = 0;

        int nn_rank = 0;
        int nnAtomInd = 0;
        
        int NooflocalNN = 0;
        int localNNs[10];

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
        
        for ( int nn_rank1 = 1 ; nn_rank1 <= d_NoofNNlist[index] ; ++nn_rank1 )
        {
            nn_rank = d_NNlist[32*index+nn_rank1-1]; // MAGIC NUMBER!!
            if ( nn_rank == cellInd ) continue;

            deltaX  = (X-d_bounding_xyz[nn_rank*6+1]>0.0f)*(X-d_bounding_xyz[nn_rank*6+1]);
            deltaX += (d_bounding_xyz[nn_rank*6+0]-X>0.0f)*(d_bounding_xyz[nn_rank*6+0]-X);

            deltaY  = (Y-d_bounding_xyz[nn_rank*6+3]>0.0f)*(Y-d_bounding_xyz[nn_rank*6+3]);
            deltaY += (d_bounding_xyz[nn_rank*6+2]-Y>0.0f)*(d_bounding_xyz[nn_rank*6+2]-Y);

            deltaZ  = (Z-d_bounding_xyz[nn_rank*6+5]>0.0f)*(Z-d_bounding_xyz[nn_rank*6+5]);
            deltaZ += (d_bounding_xyz[nn_rank*6+4]-Z>0.0f)*(d_bounding_xyz[nn_rank*6+4]-Z);

            if ( deltaX*deltaX + deltaY*deltaY + deltaZ*deltaZ > attraction_range*attraction_range )
                continue;

            ++NooflocalNN;

            if ( NooflocalNN > MAX_NN ){
                printf("Recoverable error: NooflocalNN = %d, should be < 8\n",NooflocalNN);
                continue;
            }
            localNNs[NooflocalNN-1] = nn_rank;
        }

        for ( int i = 0; i < NooflocalNN; ++i )
        {
            nn_rank =localNNs[i];

            for ( int nn_atom = 0; nn_atom < 180 ; ++nn_atom )
            {
                deltaX = X - d_X[nn_rank*192+nn_atom];
                deltaY = Y - d_Y[nn_rank*192+nn_atom];
                deltaZ = Z - d_Z[nn_rank*192+nn_atom];

                R = deltaX*deltaX + deltaY*deltaY + deltaZ*deltaZ;

                if ( R > attraction_range*attraction_range )
                    continue;

                neighVelocity = make_real3(d_velListX[nn_rank*192+nn_atom],
                                            d_velListY[nn_rank*192+nn_atom],
                                            d_velListZ[nn_rank*192+nn_atom]);

                real3 v_ij = nodeVelocity - neighVelocity;

                // Tangential component of relative velocity
                real3 vTau = v_ij - dot(v_ij, normal)*normal;
                force = force - gamma_ext*vTau;
            }

        }

        // viscous drag
        force = force - gamma_o*nodeVelocity;
        
        // write force to global memory
        d_fDisList.x[globalNodeInd] = force.x; 
        d_fDisList.y[globalNodeInd] = force.y; 
        d_fDisList.z[globalNodeInd] = force.z; 
    }
}


__global__ void Integrate(real *d_XP, real *d_YP, real *d_ZP,
                          real *d_X, real *d_Y, real *d_Z, 
                          real *d_XM, real *d_YM, real *d_ZM,
                          real *d_velListX, real *d_velListY, real *d_velListZ, 
                          real *d_time, real m,
                          R3Nptrs d_fConList, R3Nptrs d_fDisList, R3Nptrs d_fRanList,
                          int numCells, bool add_rands,
                          curandState *rngStates, real rand_scale_factor){
    const int cellInd = blockIdx.x;
    const int node = threadIdx.x;
    
    
    if (cellInd < numCells && node < 180){
        int nodeInd = cellInd*192 + node;
        const real dt = d_time[0];
        const real root_dt = sqrtf(dt);
        
        d_velListX[nodeInd] = d_velListX[nodeInd] + 0.5*(dt*d_fConList.x[nodeInd] + dt*d_fDisList.x[nodeInd] + \
                                                         root_dt*d_fRanList.x[nodeInd])/m;
        
        d_velListY[nodeInd] = d_velListY[nodeInd] + 0.5*(dt*d_fConList.y[nodeInd] + dt*d_fDisList.y[nodeInd] + \
                                                         root_dt*d_fRanList.y[nodeInd])/m;
        
        d_velListZ[nodeInd] = d_velListZ[nodeInd] + 0.5*(dt*d_fConList.z[nodeInd] + dt*d_fDisList.z[nodeInd] + \
                                                         root_dt*d_fRanList.z[nodeInd])/m;

        d_XP[nodeInd] = d_X[nodeInd] + d_velListX[nodeInd]*dt; 
        d_YP[nodeInd] = d_Y[nodeInd] + d_velListY[nodeInd]*dt; 
        d_ZP[nodeInd] = d_Z[nodeInd] + d_velListZ[nodeInd]*dt; 

        if (add_rands != 0){
            curandState rngState = rngStates[nodeInd];
            real3 r = rand_scale_factor*make_real3(curand_uniform(&rngState) - 0.5f,
                                                     curand_uniform(&rngState) - 0.5f,
                                                     curand_uniform(&rngState) - 0.5f);

            d_XP[nodeInd] += r.x; 
            d_YP[nodeInd] += r.y; 
            d_ZP[nodeInd] += r.z; 
            
            rngStates[nodeInd] = rngState;
        }
    }
}

__global__ void VelocityUpdateA(real* d_VX, real* d_VY, real* d_VZ,
                                R3Nptrs fConList, R3Nptrs fRanList,
                                real dt, long int num_nodes, real m){
    long int nodeInd = blockIdx.x*blockDim.x + threadIdx.x;

    if (nodeInd < num_nodes){
        real root_dt = sqrtf(dt);
        d_VX[nodeInd] = d_VX[nodeInd] + 0.5*(dt*fConList.x[nodeInd] + root_dt*fRanList.x[nodeInd])/m;
        d_VY[nodeInd] = d_VY[nodeInd] + 0.5*(dt*fConList.y[nodeInd] + root_dt*fRanList.y[nodeInd])/m;
        d_VZ[nodeInd] = d_VZ[nodeInd] + 0.5*(dt*fConList.z[nodeInd] + root_dt*fRanList.z[nodeInd])/m;
    }
}

__global__ void VelocityUpdateB(real* d_VX, real* d_VY, real* d_VZ,
                                R3Nptrs fDisList, real dt, long int num_nodes, real m){
    long int nodeInd = blockIdx.x*blockDim.x + threadIdx.x;

    if (nodeInd < num_nodes){
        d_VX[nodeInd] = d_VX[nodeInd] + 0.5*dt*(fDisList.x[nodeInd])/m;
        d_VY[nodeInd] = d_VY[nodeInd] + 0.5*dt*(fDisList.y[nodeInd])/m;
        d_VZ[nodeInd] = d_VZ[nodeInd] + 0.5*dt*(fDisList.z[nodeInd])/m;
    }
}

__global__ void ForwardTime(real *d_XP, real *d_YP, real *d_ZP,
                            real *d_X, real *d_Y, real *d_Z,
                            real *d_XM, real *d_YM, real *d_ZM, 
                            int numCells){
    
    const int nodeInd = blockIdx.x*blockDim.x + threadIdx.x;
    if (nodeInd < 192*numCells){
        // if (d_XP[nodeInd] != d_XM[nodeInd] ||
        //     d_YP[nodeInd] != d_YM[nodeInd] || 
        //     d_ZP[nodeInd] != d_ZM[nodeInd] ){
        //     printf("%.20f != %.20f or\n%.20f != %.20f or\n%.20f != %.20f\nnodeInd=%d\n",
        //            d_XP[nodeInd], d_XM[nodeInd],
        //            d_YP[nodeInd], d_YM[nodeInd],
        //            d_ZP[nodeInd], d_ZM[nodeInd], nodeInd);
        //     asm("trap;");
        // }

        d_XM[nodeInd] = d_X[nodeInd]; 
        d_YM[nodeInd] = d_Y[nodeInd]; 
        d_ZM[nodeInd] = d_Z[nodeInd];
    
        d_X[nodeInd] = d_XP[nodeInd];
        d_Y[nodeInd] = d_YP[nodeInd];
        d_Z[nodeInd] = d_ZP[nodeInd];
    }
}


__global__ void CorrectCoMMotion(SimStatePtrs sim_state, real3 sysCM){
    
    long int partInd = blockIdx.x*blockDim.x + threadIdx.x;

    if (partInd < sim_state.no_of_cells*192){
        sim_state.posP.x[partInd] -= sysCM.x;
        sim_state.posP.y[partInd] -= sysCM.y;
        sim_state.posP.z[partInd] -= sysCM.z;
        
        sim_state.pos.x[partInd] -= sysCM.x;
        sim_state.pos.y[partInd] -= sysCM.y;
        sim_state.pos.z[partInd] -= sysCM.z;
        
        sim_state.posM.x[partInd] -= sysCM.x;
        sim_state.posM.y[partInd] -= sysCM.y;
        sim_state.posM.z[partInd] -= sysCM.z;
    }
}


__global__ void SumForces(R3Nptrs fConList, R3Nptrs fDisList, R3Nptrs fRanList,
                          R3Nptrs fList, 
                          long int numNodes){
    size_t idx = blockIdx.x*blockDim.x + threadIdx.x;
    if (idx < numNodes){
        fList.x[idx] = fConList.x[idx] + fDisList.x[idx] + fRanList.x[idx]; 
        fList.y[idx] = fConList.y[idx] + fDisList.y[idx] + fRanList.y[idx]; 
        fList.z[idx] = fConList.z[idx] + fDisList.z[idx] + fRanList.z[idx];
    }
}

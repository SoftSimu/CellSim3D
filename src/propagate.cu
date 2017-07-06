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
                                      angles3 d_theta0[], real k, int cellInd){
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

    __shared__ real* d_X; 
    __shared__ real* d_Y; 
    __shared__ real* d_Z;
    __shared__ int* d_C180_nn;
    __shared__ real* d_R0;
    __shared__ real3 cellCOM;
    __shared__ real Pressure;
    __shared__ real stiffness;
    __shared__ real* d_bounding_xyz;
    __shared__ real attr_range;
    __shared__ real attr_stren;
    __shared__ real rep_range;
    __shared__ real rep_stren;
    __shared__ real thresh_dist;
    __shared__ real3 box_max;
    __shared__ real* d_Fx; 
    __shared__ real* d_Fy; 
    __shared__ real* d_Fz;
    __shared__ angles3* d_theta0;
    __shared__ real DL;
    __shared__ int* d_numOfNNList;
    __shared__ int* d_nnList;
    size_t cellInd, localNodeInd;
    
    cellInd = blockIdx.x;
    localNodeInd = threadIdx.x;
    
    size_t cellOffset = cellInd*192;
    size_t nodeInd = cellOffset + localNodeInd;
    if ( cellInd < simState.no_of_cells && localNodeInd < 180 )
    {
        if (threadIdx.x == 0){
            d_X = simState.pos.x; 
            d_Y = simState.pos.y; 
            d_Z = simState.pos.z;
            d_C180_nn = simState.C180_nn;
            d_R0 = simState.R0;
            Pressure = simState.pressures[cellInd];
            stiffness = simState.youngsMod[cellInd];
            
            cellCOM = make_real3(simState.cellCOMs.x[cellInd],
                                 simState.cellCOMs.y[cellInd],
                                 simState.cellCOMs.z[cellInd]);
            
            attr_range = sim_params.core_params.attr_range;
            attr_stren = sim_params.core_params.attr_stiff;
            
            rep_range = sim_params.core_params.rep_range;
            rep_stren = sim_params.core_params.rep_stiff;
            thresh_dist = sim_params.box_params.thresh_dist;
            box_max = sim_params.box_params.box_max;
            d_Fx = simState.conForce.x; 
            d_Fy = simState.conForce.y; 
            d_Fz = simState.conForce.z;
            d_theta0 = simState.theta0;
            DL = sim_params.core_params.dom_len;
            d_numOfNNList = simState.numOfNNList;
            d_nnList = simState.nnList;
        }
        __syncthreads();
        
        real3 nodePos = make_real3(d_X[nodeInd],
                                   d_Y[nodeInd],
                                   d_Z[nodeInd]);
        real3 force = make_real3(0, 0, 0); 
        
        if (!good_real3(nodePos)){
            printf("OH SHIT: we have a nan\n");
            printf("Particle index: %d, Cell: %d\n", localNodeInd, cellInd);
            printf("Crash now :(\n"); 
            asm("trap;"); 
        }

        //  Spring Force calculation within cell
        //  go through three nearest neighbors
        for ( int i = 0; i < 3 ; ++i ) // Maybe Better to open this loop
        {
            int N1 = d_C180_nn[i*192+localNodeInd];

            real3 delta = make_real3(d_X[cellInd*192+N1],
                                     d_Y[cellInd*192+N1],
                                     d_Z[cellInd*192+N1]) - nodePos;
            real R = mag(delta);

            real R0 = d_R0[i*192 + localNodeInd];

            //spring forces
            force = force + (stiffness*(R-R0)/R)*delta;
        }

        // new growth force
        real3 rCM = nodePos - cellCOM;
        
        force = force + 3*Pressure*calcUnitVec(rCM); //why the 3? Don't know:)

        //gForce = -10*(volList[cellInd] - div_vol)*calcUnitVec(r_CM);
        //gForce = -10*(mag(r_CM) - r_CM_o)*calcUnitVec(r_CM);
        
        if (sim_params.angle_params.angle_pot){
            real3 t = CalculateAngleForce(localNodeInd, d_C180_nn,
                                          d_X, d_Y, d_Z,
                                          d_theta0, stiffness, cellInd);
            force = force + t;
        }
        
        
        // interfullerene attraction and repulsion
        
        int NooflocalNN = 0;

        int startx = (int)((nodePos.x - simState.mins.x[0])/DL);
        if ( startx < 0 ) startx = 0;
        if ( startx >= Xdiv ) startx = Xdiv-1;

        int starty = (int)((nodePos.y - simState.mins.y[0])/DL);
        if ( starty < 0 ) starty = 0;
        if ( starty >= Ydiv ) starty = Ydiv-1;

        int startz = (int)((nodePos.z - simState.mins.z[0])/DL);
        if ( startz < 0 ) startz = 0;
        if ( startz >= Zdiv ) startz = Zdiv-1;

        int index = startz*Xdiv*Ydiv + starty*Xdiv + startx;
        

        int nn_CellInd;
        real deltaX, deltaY, deltaZ;
        int localNNs[MAX_NN];
        for ( int nn_rank1 = 1 ; nn_rank1 <= d_numOfNNList[index] ; ++nn_rank1 ){
            nn_CellInd = d_nnList[32*index+nn_rank1-1];
            
            if ( nn_CellInd == cellInd )
                continue;

            deltaX  = (nodePos.x-d_bounding_xyz[nn_CellInd*6+1]>0.0f)*(nodePos.x-d_bounding_xyz[nn_CellInd*6+1]);
            deltaX += (d_bounding_xyz[nn_CellInd*6+0]-nodePos.x>0.0f)*(d_bounding_xyz[nn_CellInd*6+0]-nodePos.x);

            deltaY  = (nodePos.y-d_bounding_xyz[nn_CellInd*6+3]>0.0f)*(nodePos.y-d_bounding_xyz[nn_CellInd*6+3]);
            deltaY += (d_bounding_xyz[nn_CellInd*6+2]-nodePos.y>0.0f)*(d_bounding_xyz[nn_CellInd*6+2]-nodePos.y);

            deltaZ  = (nodePos.z-d_bounding_xyz[nn_CellInd*6+5]>0.0f)*(nodePos.z-d_bounding_xyz[nn_CellInd*6+5]);
            deltaZ += (d_bounding_xyz[nn_CellInd*6+4]-nodePos.z>0.0f)*(d_bounding_xyz[nn_CellInd*6+4]-nodePos.z);

            if (deltaX*deltaX + deltaY*deltaY + deltaZ*deltaZ >
                attr_range*attr_range)
                continue;

            ++NooflocalNN;

            //printf("NooflocalNN %d\n", NooflocalNN);

            if ( NooflocalNN > MAX_NN ){
                printf("Recoverable error: NooflocalNN = %d, should be < 8\n", NooflocalNN);
                continue;
            }

            localNNs[NooflocalNN-1] = nn_CellInd;
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

                real R = mag2(Rvec);
                if (R > attr_range*attr_range)
                    continue;

                R = sqrtf(R);
                
                contactForce = -attr_stren*stiffness*(attr_range-R)*calcUnitVec(Rvec);
                
                if (R <= rep_range){
                    //if (R < (rep_range-0.01)) R = repulsion_range-0.01;
                                        
                    contactForce = contactForce +
                        rep_stren*stiffness*(rep_range-R)*calcUnitVec(Rvec);
                }
            }

        }

        // add forces from simulation box if needed:
        if (sim_params.box_params.use_rigid_sim_box){
            real gap1, gap2; 

            gap1 = nodePos.x; 
            gap2 = box_max.x - nodePos.x; 

            if (gap1 < thresh_dist){
                force.x += -rep_stren*stiffness*(gap1-thresh_dist);
            }

            if (gap2 < thresh_dist){
                force.x += rep_stren*stiffness*(gap2 - thresh_dist);
            }

            gap1 = nodePos.y; 
            gap2 = box_max.y - nodePos.y; 

            if (gap1 < thresh_dist){
                force.y += -rep_stren*stiffness*(gap1-thresh_dist);
            }

            if (gap2 < thresh_dist){
                force.y += rep_stren*stiffness*(gap2 - thresh_dist);
            }
            
            gap1 = nodePos.z; 
            gap2 = box_max.z - nodePos.z; 

            if (gap1 < thresh_dist){
                force.z += -rep_stren*stiffness*(gap1-thresh_dist);
            }

            if (gap2 < thresh_dist){
                force.z += rep_stren*stiffness*(gap2 - thresh_dist);
            }
            
        }

        d_Fx[nodeInd] = force.x;
        d_Fy[nodeInd] = force.y;
        d_Fz[nodeInd] = force.z;
    }
}


__global__ void CalculateDisForce( SimStatePtrs sim_state, sim_params_struct sim_params,
                                   int Xdiv, int Ydiv, int Zdiv){

    __shared__ real* d_X;
    __shared__ real* d_Y;
    __shared__ real* d_Z;
    __shared__ real* d_VX;
    __shared__ real* d_VY;
    __shared__ real* d_VZ;
    
    __shared__ real* d_FX;
    __shared__ real* d_FY;
    __shared__ real* d_FZ;
    
    __shared__ int no_of_cells;
    __shared__ int* d_C180_nn;
    __shared__ int* d_C180_sign;
    __shared__ real attr_range;
    __shared__ real DL;
    __shared__ int* d_NNlist;
    __shared__ int* d_NoofNNlist;
    __shared__ real* d_bounding_xyz;
    __shared__ real gamma_int;
    __shared__ real gamma_ext;
    __shared__ real gamma_o;

    if (threadIdx.x == 0){
        d_X = sim_state.pos.x;
        d_Y = sim_state.pos.y;
        d_Z = sim_state.pos.z;
        d_VX = sim_state.vel.x;
        d_VY = sim_state.vel.y;
        d_VZ = sim_state.vel.z;
        no_of_cells = sim_state.no_of_cells;
        d_C180_nn = sim_state.C180_nn;
        d_C180_sign = sim_state.C180_sign;
        attr_range = sim_params.core_params.attr_range;
        DL = sim_params.core_params.dom_len;
        d_NNlist = sim_state.nnList;
        d_NoofNNlist = sim_state.numOfNNList;
        d_bounding_xyz = sim_state.boundingBoxes;
        gamma_int = sim_params.core_params.internal_damping;
        gamma_ext = sim_params.core_params.inter_membr_fric;
        gamma_o = sim_params.core_params.gamma_visc;
        d_FX = sim_state.disForce.x;
        d_FY = sim_state.disForce.y;
        d_FZ = sim_state.disForce.z;
    }
    __syncthreads();
    
    size_t cellInd = blockIdx.x;
    size_t nodeInd = threadIdx.x;

    if (cellInd < no_of_cells && nodeInd < 180){
        size_t globalNodeInd = cellInd*192 + nodeInd;
        size_t N = 0;
        real3 force = make_real3(0, 0, 0);
        
        real3 nodeVelocity = make_real3(d_VX[globalNodeInd],
                                        d_VY[globalNodeInd],
                                        d_VZ[globalNodeInd]);

        real3 neighVelocity;
        
        // dampen bonding
        for (int i = 0; i < 3; ++i){
            N = d_C180_nn[i*192+nodeInd];
            neighVelocity = make_real3(d_VX[cellInd*192+N],
                                       d_VY[cellInd*192+N],
                                       d_VZ[cellInd*192+N]);
                
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

        real3 nodePos = make_real3(d_X[globalNodeInd],
                                   d_Y[globalNodeInd],
                                   d_Z[globalNodeInd]);

        real deltaX = 0;
        real deltaY = 0;
        real deltaZ = 0;
        
        int nn_CellInd = 0;
        
        int NooflocalNN = 0;

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
        int localNNs[MAX_NN];
        for ( int nn_rank1 = 1 ; nn_rank1 <= d_NoofNNlist[index] ; ++nn_rank1 )
        {
            nn_CellInd = d_NNlist[32*index+nn_rank1-1]; // MAGIC NUMBER!!
            if ( nn_CellInd == cellInd ) continue;

            deltaX  = (nodePos.x-d_bounding_xyz[nn_CellInd*6+1]>0.0f)*(nodePos.x-d_bounding_xyz[nn_CellInd*6+1]);
            deltaX += (d_bounding_xyz[nn_CellInd*6+0]-nodePos.x>0.0f)*(d_bounding_xyz[nn_CellInd*6+0]-nodePos.x);

            deltaY  = (nodePos.y-d_bounding_xyz[nn_CellInd*6+3]>0.0f)*(nodePos.y-d_bounding_xyz[nn_CellInd*6+3]);
            deltaY += (d_bounding_xyz[nn_CellInd*6+2]-nodePos.y>0.0f)*(d_bounding_xyz[nn_CellInd*6+2]-nodePos.y);

            deltaZ  = (nodePos.z-d_bounding_xyz[nn_CellInd*6+5]>0.0f)*(nodePos.z-d_bounding_xyz[nn_CellInd*6+5]);
            deltaZ += (d_bounding_xyz[nn_CellInd*6+4]-nodePos.z>0.0f)*(d_bounding_xyz[nn_CellInd*6+4]-nodePos.z);

            if ( deltaX*deltaX + deltaY*deltaY + deltaZ*deltaZ > attr_range*attr_range)
                continue;

            ++NooflocalNN;

            if ( NooflocalNN > MAX_NN ){
                printf("Recoverable error: NooflocalNN = %d, should be < 8\n",NooflocalNN);
                continue;
            }
            localNNs[NooflocalNN-1] = nn_CellInd;
        }


        for ( int i = 0; i < NooflocalNN; ++i )
        {
            nn_CellInd =localNNs[i];
            for ( int nn_atom = 0; nn_atom < 180 ; ++nn_atom )
            {
                int nnAtomInd = nn_CellInd*192+nn_atom;
                real3 delta = nodePos - make_real3(d_X[nnAtomInd],
                                                   d_Y[nnAtomInd],
                                                   d_Z[nnAtomInd]);

                if (mag2(delta) > attr_range*attr_range)
                    continue;

                neighVelocity = make_real3(d_VX[nn_CellInd*192+nn_atom],
                                           d_VY[nn_CellInd*192+nn_atom],
                                           d_VZ[nn_CellInd*192+nn_atom]);

                real3 v_ij = nodeVelocity - neighVelocity;

                // Tangential component of relative velocity
                real3 vTau = v_ij - dot(v_ij, normal)*normal;
                force = force - gamma_ext*vTau;
            }
        }

        // viscous drag
        force = force - gamma_o*nodeVelocity;
        
        // write force to global memory
        d_FX[globalNodeInd] = force.x; 
        d_FY[globalNodeInd] = force.y; 
        d_FZ[globalNodeInd] = force.z; 
    }
}


__global__ void Integrate(SimStatePtrs sim_state, sim_params_struct sim_params){
    
    __shared__ real *d_XP;
    __shared__ real *d_YP;
    __shared__ real *d_ZP;
    
    __shared__ real *d_X;
    __shared__ real *d_Y;
    __shared__ real *d_Z;
    
    __shared__ real *d_VX;
    __shared__ real *d_VY; 
    __shared__ real *d_VZ;

    __shared__ real *d_FCX; 
    __shared__ real *d_FCY; 
    __shared__ real *d_FCZ; 

    __shared__ real *d_FDX; 
    __shared__ real *d_FDY; 
    __shared__ real *d_FDZ; 

    __shared__ real *d_FRX;
    __shared__ real *d_FRY; 
    __shared__ real *d_FRZ;
    __shared__ real m;
    __shared__ long int numCells;
    __shared__ real dt;
    __shared__ real root_dt; 
    
    const int cellInd = blockIdx.x;
    const int node = threadIdx.x;


    if (threadIdx.x == 0){
        d_XP = sim_state.posP.x;
        d_YP = sim_state.posP.y;
        d_ZP = sim_state.posP.z;
            
        d_X = sim_state.pos.x;
        d_Y = sim_state.pos.y;
        d_Z = sim_state.pos.z;

        d_FCX = sim_state.conForce.x;
        d_FCY = sim_state.conForce.y;
        d_FCZ = sim_state.conForce.z;
             
        d_FDX = sim_state.disForce.x;
        d_FDY = sim_state.disForce.y;
        d_FDZ = sim_state.disForce.z;
             
        d_FRX = sim_state.ranForce.x;
        d_FRY = sim_state.ranForce.y;
        d_FRZ = sim_state.ranForce.z;
             
        d_VX = sim_state.vel.x;
        d_VY = sim_state.vel.y;
        d_VZ = sim_state.vel.z;

        m = sim_params.core_params.node_mass;
        numCells = sim_state.no_of_cells;
        dt = sim_params.core_params.delta_t;
        root_dt = sqrtf(dt); 
    }
    
    
    if (cellInd < numCells && node < 180){
        int nodeInd = cellInd*192 + node;
        
        real3 vel = make_real3 (d_VX[nodeInd],
                                d_VY[nodeInd],
                                d_VZ[nodeInd]);
        
        real3 conForce = make_real3(d_FCX[nodeInd],
                                    d_FCY[nodeInd],
                                    d_FCZ[nodeInd]);

        real3 disForce = make_real3(d_FDX[nodeInd],
                                    d_FDY[nodeInd],
                                    d_FDZ[nodeInd]); // not datForce

        real3 ranForce = make_real3(d_FRX[nodeInd],
                                    d_FRY[nodeInd],
                                    d_FRZ[nodeInd]);
                                    

        real3 posP = make_real3(0,0,0);

        real3 pos = make_real3(d_X[nodeInd],
                               d_Y[nodeInd],
                               d_Z[nodeInd]);
        

        vel = vel + 0.5*(dt*conForce + dt*disForce + root_dt*ranForce)/m;

        posP = pos + dt*vel;

        d_VX[nodeInd] = vel.x;
        d_VY[nodeInd] = vel.y;
        d_VZ[nodeInd] = vel.z;

        d_XP[nodeInd] = posP.x; 
        d_YP[nodeInd] = posP.y; 
        d_ZP[nodeInd] = posP.z; 
    }
}

__global__ void VelocityUpdateA(SimStatePtrs sim_state, sim_params_struct sim_params){
    

    __shared__ real *d_VX;
    __shared__ real *d_VY;
    __shared__ real *d_VZ;
    __shared__ real *d_fConX;
    __shared__ real *d_fConY;
    __shared__ real *d_fConZ;
    __shared__ real *d_fRanX;
    __shared__ real *d_fRanY;
    __shared__ real *d_fRanZ;
    __shared__ real m;
    __shared__ real dt;
    __shared__ long int num_nodes; 

    if (threadIdx.x == 0){
        d_VX = sim_state.vel.x;
        d_VY = sim_state.vel.y;
        d_VZ = sim_state.vel.z;

        d_fConX = sim_state.conForce.x; 
        d_fConY = sim_state.conForce.y; 
        d_fConZ = sim_state.conForce.z; 

        d_fRanX = sim_state.ranForce.x; 
        d_fRanY = sim_state.ranForce.y; 
        d_fRanZ = sim_state.ranForce.z;

        m = sim_params.core_params.node_mass;
        dt = sim_params.core_params.delta_t;
        num_nodes = sim_state.no_of_cells*192;  // hard-coded :(
        
    }
    __syncthreads();

    long int nodeInd = blockIdx.x*blockDim.x + threadIdx.x;

    if (nodeInd < num_nodes){
        real root_dt = sqrtf(dt);
        d_VX[nodeInd] = d_VX[nodeInd] + 0.5*(dt*d_fConX[nodeInd] + root_dt*d_fRanX[nodeInd])/m;
        d_VY[nodeInd] = d_VY[nodeInd] + 0.5*(dt*d_fConY[nodeInd] + root_dt*d_fRanY[nodeInd])/m;
        d_VZ[nodeInd] = d_VZ[nodeInd] + 0.5*(dt*d_fConZ[nodeInd] + root_dt*d_fRanZ[nodeInd])/m;
    }
}

__global__ void VelocityUpdateB(SimStatePtrs sim_state, sim_params_struct sim_params){
    __shared__ real *d_VX;
    __shared__ real *d_VY;
    __shared__ real *d_VZ;
    __shared__ real *d_fDisX; 
    __shared__ real *d_fDisY; 
    __shared__ real *d_fDisZ; 
    __shared__ real m;
    __shared__ real dt;
    __shared__ long int num_nodes; 

    if (threadIdx.x == 0){
        d_VX = sim_state.vel.x;
        d_VY = sim_state.vel.y;
        d_VZ = sim_state.vel.z;
        d_fDisX = sim_state.disForce.x;
        d_fDisY = sim_state.disForce.y;
        d_fDisZ = sim_state.disForce.z;
        m = sim_params.core_params.node_mass;
        dt = sim_params.core_params.delta_t;
        num_nodes = sim_state.no_of_cells*192;
    }
    
    long int nodeInd = blockIdx.x*blockDim.x + threadIdx.x;

    if (nodeInd < num_nodes){
        d_VX[nodeInd] = d_VX[nodeInd] + 0.5*dt*(d_fDisX[nodeInd])/m;
        d_VY[nodeInd] = d_VY[nodeInd] + 0.5*dt*(d_fDisY[nodeInd])/m;
        d_VZ[nodeInd] = d_VZ[nodeInd] + 0.5*dt*(d_fDisZ[nodeInd])/m;
    }
}

__global__ void ForwardTime(SimStatePtrs sim_state){

    __shared__  real *d_XP;
    __shared__  real *d_YP;
    __shared__  real *d_ZP;
    __shared__  real *d_X;
    __shared__  real *d_Y;
    __shared__  real *d_Z;
    __shared__  real *d_XM;
    __shared__  real *d_YM;
    __shared__  real *d_ZM;
    __shared__ int no_of_cells; 

    
    const int nodeInd = blockIdx.x*blockDim.x + threadIdx.x;

    if (threadIdx.x == 0){
        d_XP = sim_state.posP.x;
        d_YP = sim_state.posP.y;
        d_ZP = sim_state.posP.z;
        
        d_X = sim_state.pos.x;
        d_Y = sim_state.pos.y;
        d_Z = sim_state.pos.z;
        
        d_XM = sim_state.posM.x;
        d_YM = sim_state.posM.y;
        d_ZM = sim_state.posM.z;
        no_of_cells = sim_state.no_of_cells; 
    }
    
    if (nodeInd < 192*no_of_cells){
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

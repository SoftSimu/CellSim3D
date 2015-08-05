#include<stdio.h>
__device__ void PositionUpdate(float Vx, float Vy, float Vz,
                               float Ax, float Ay, float Az,
                               float delta_t, int index, 
                               // outputs
                               float* X, float* Y, float* Z) {
    
    X[index] = X[index] + Vx*delta_t + 0.5*Ax*delta_t*delta_t; 
    Y[index] = Y[index] + Vy*delta_t + 0.5*Ay*delta_t*delta_t; 
    Z[index] = Z[index] + Vz*delta_t + 0.5*Az*delta_t*delta_t; 
}



// This will mostly be used at half time step increments.
// In practise dt should= 0.5*delta_t
__device__ void VelocityUpdate(float Ax, float Ay, float Az,
                               float dt,
                               float VxIn, float VyIn, float VzIn, 
                               // Outputs: 
                               float* Vx, float* Vy, float* Vz){
                               
    *Vx = VxIn + Ax*dt; 
    *Vy = VyIn + Ay*dt; 
    *Vz = VzIn + Az*dt; 
}


// Now for the force calculation kernel
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
    int cellOffset = cellInd*192;
    int globalInd = cellOffset+nodeInd; 
    float deltaX = 0;
    float deltaY = 0;
    float deltaZ = 0;

    float Fx = 0;
    float Fy = 0;
    float Fz = 0;

    // Calculate normal to the surface at this node

    int N1 = d_C180_nn[0   + nodeInd];
    int N2 = d_C180_nn[192 + nodeInd];
    int N3 = d_C180_nn[384 + nodeInd];

    float A1 = d_X[cellOffset + N2] - d_X[cellOffset + N1];
    float A2 = d_Y[cellOffset + N2] - d_Y[cellOffset + N1];
    float A3 = d_Z[cellOffset + N2] - d_Z[cellOffset + N1];

    float B1 = d_X[cellOffset + N3] - d_X[cellOffset + N1];
    float B2 = d_Y[cellOffset + N3] - d_Y[cellOffset + N1];
    float B3 = d_Z[cellOffset + N3] - d_Z[cellOffset + N1];

    float TX = A2*B3-A3*B2;
    float TY = A3*B1-A1*B3;
    float TZ = A1*B2-A2*B1;

    float norm = sqrt(TX*TX + TY*TY + TZ*TZ);

    float nX = d_C180_sign[nodeInd]*TX/norm; 
    float nY = d_C180_sign[nodeInd]*TY/norm; 
    float nZ = d_C180_sign[nodeInd]*TZ/norm; 
        
    
    float X = d_X[globalInd]; 
    float Y = d_Y[globalInd]; 
    float Z = d_Z[globalInd];

    float Vx = VxL[globalInd];
    float Vy = VyL[globalInd];
    float Vz = VzL[globalInd];
    
    float R = 0;
    int neighInd = -2;
    float pressure = pressList[globalInd];
    
    // Calculate forces over all the bonded neighbours (3 in this case)
    for (int b = 0; b < 3; b++){
        neighInd = d_C180_nn[b*192 + nodeInd];
        
        deltaX = d_X[cellOffset+neighInd] - X;
        deltaY = d_Y[cellOffset+neighInd] - Y;
        deltaZ = d_Z[cellOffset+neighInd] - Z;

        R = sqrt(deltaX*deltaX + deltaY*deltaY + deltaZ*deltaZ); 


        // Bonding force
        Fx += Youngs_mod*(R-R0)/R0*deltaX/R;
        Fy += Youngs_mod*(R-R0)/R0*deltaY/R;
        Fz += Youngs_mod*(R-R0)/R0*deltaZ/R;

        // Bonding force damping
        Fx += -gamma_int*(Vx - VxL[cellOffset+neighInd]);
        Fy += -gamma_int*(Vy - VyL[cellOffset+neighInd]);
        Fz += -gamma_int*(Vz - VzL[cellOffset+neighInd]);

        // Pressure force
        Fx += pressure*nX;
        Fy += pressure*nY;
        Fz += pressure*nZ;
    }

    // Intercellular forces
    int noLocalNeigh = 0;
    int localNNs[10]; 
    int startX = (int)((X - minX)/DL); 
    if (startX < 0) startX = 0;
    if (startX >= xDiv) startX = xDiv - 1;

    int startY = (int)((Y - minY)/DL); 
    if (startY < 0) startY = 0;
    if (startY >= yDiv) startY = yDiv - 1;

    int startZ = (int)((Z - minZ)/DL); 
    if (startZ < 0) startZ = 0;
    if (startZ >= zDiv) startZ = zDiv - 1;

    int index = startZ*xDiv*yDiv + startY*xDiv + startX;
    int nnCellInd = -1; 
    for (int nn_rank1 = 1; nn_rank1 <= d_NoofNNlist[index]; ++nn_rank1){
        nnCellInd = d_NNlist[32*index+nn_rank1-1];

        if (nnCellInd == cellInd) continue;

        deltaX  = (X-d_bounding_xyz[nnCellInd*6+1]>0.0f)*(X-d_bounding_xyz[nnCellInd*6+1]);
        deltaX += (d_bounding_xyz[nnCellInd*6+0]-X>0.0f)*(d_bounding_xyz[nnCellInd*6+0]-X);

        deltaY  = (Y-d_bounding_xyz[nnCellInd*6+3]>0.0f)*(Y-d_bounding_xyz[nnCellInd*6+3]);
        deltaY += (d_bounding_xyz[nnCellInd*6+2]-Y>0.0f)*(d_bounding_xyz[nnCellInd*6+2]-Y);

        deltaZ  = (Z-d_bounding_xyz[nnCellInd*6+5]>0.0f)*(Z-d_bounding_xyz[nnCellInd*6+5]);
        deltaZ += (d_bounding_xyz[nnCellInd*6+4]-Z>0.0f)*(d_bounding_xyz[nnCellInd*6+4]-Z);

        if (deltaX*deltaX + deltaY*deltaY + deltaZ*deltaZ > attraction_range*attraction_range)
            continue;

        ++noLocalNeigh;
        
        if ( noLocalNeigh > 10 ){
            printf("Recoverable error: NooflocalNN = %d, should be < 8\n",noLocalNeigh);
            continue;
        }
        localNNs[noLocalNeigh-1]= nnCellInd*192; 
    }

    for (int i = 0; i < noLocalNeigh; ++i){
        nnCellInd = localNNs[i];

        for (int nnNodeInd = 0; nnNodeInd < 180; ++nnNodeInd){
            deltaX = X - d_X[nnCellInd+nnNodeInd];
            deltaY = Y - d_Y[nnCellInd+nnNodeInd];
            deltaZ = Z - d_Z[nnCellInd+nnNodeInd];

            R = sqrt(deltaX*deltaX + deltaY*deltaY + deltaZ*deltaZ);

            if (R >=  attraction_range)
                continue;

            // Attraction force
            Fx += -attraction_strength*Youngs_mod*(attraction_range-R)/R*deltaX;
            Fy += -attraction_strength*Youngs_mod*(attraction_range-R)/R*deltaY;
            Fz += -attraction_strength*Youngs_mod*(attraction_range-R)/R*deltaZ;

            // Intercellular friction
            // Calculate relative velocity
            float VijX = Vx - VxL[nnCellInd+nnNodeInd];
            float VijY = Vy - VyL[nnCellInd+nnNodeInd];
            float VijZ = Vz - VzL[nnCellInd+nnNodeInd];

            // Dot product between relative velocity and normal to surace
            float VijDotN = VijX*nX + VijY*nY + VijZ*nZ;

            // Tangential component
            float vTauX = VijX - VijDotN*nX; 
            float vTauY = VijY - VijDotN*nY; 
            float vTauZ = VijZ - VijDotN*nZ;

            // Force is proportional to vTau
            Fx += -gamma_int*vTauX; 
            Fy += -gamma_int*vTauY; 
            Fz += -gamma_int*vTauZ;

            if (R < repulsion_range){
                if (R < (repulsion_range-0.01)) R = repulsion_range-0.01;

                Fx += repulsion_strength*Youngs_mod*(repulsion_range-R)/R*deltaX; 
                Fy += repulsion_strength*Youngs_mod*(repulsion_range-R)/R*deltaY; 
                Fz += repulsion_strength*Youngs_mod*(repulsion_range-R)/R*deltaZ;

            }
            
        }
    }

    // Viscous drag force
    Fx += -gamma_o*Vx; 
    Fy += -gamma_o*Vy; 
    Fz += -gamma_o*Vz;

    // Add wall force if needed

    if (useWalls == 1){
        float wallDist = Z - wall1; 
        if (abs(wallDist) <= threshDist){
            if (wallDist*Fz < 0) Fz = -Fz;
        }

        wallDist = Z - wall2;
        if (abs(wallDist) <= threshDist){
            if (wallDist*Fz < 0) Fz = -Fz;
        }
    }

    FxL[globalInd] = Fx;
    FyL[globalInd] = Fy;
    FzL[globalInd] = Fz;
    
}

                               
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
                          float threshDist){

    int cellInd = blockIdx.x;
    int nodeInd = threadIdx.x;
    int globalInd = cellInd*192+nodeInd; 
    
    if (cellInd < No_of_C180s && nodeInd < 180){
        
        // Calculate position at next time step
        PositionUpdate(VxL[globalInd], VyL[globalInd], VzL[globalInd],
                       FxL[globalInd]/m, FyL[globalInd]/m, FzL[globalInd]/m,
                       delta_t, globalInd, 
                       d_X, d_Y, d_Z);
        
        // Calculate velocity at half time step
        float vHalfX = 0;
        float vHalfY = 0;
        float vHalfZ = 0; 
                       
        VelocityUpdate(FxL[globalInd]/m, FyL[globalInd]/m, FzL[globalInd]/m,
                       0.5*delta_t,
                       VxL[globalInd], VyL[globalInd], VzL[globalInd], 
                       &vHalfX, &vHalfY, &vHalfZ);

        // Calculate Forces for new positions
        CalculateForce(nodeInd, cellInd, m,
                       internal_damping, delta_t, d_pressList,
                       Youngs_mod, R0, attraction_strength,
                       attraction_range, repulsion_strength,
                       repulsion_range, viscotic_damping,
                       gamma_visc, DL,
                       Minx, Miny, Minz,
                       xDiv, yDiv, zDiv,
                       VxL, VyL, VzL,
                       d_C180_nn, d_C180_sign, d_NoofNNlist,
                       d_NNlist, d_bounding_xyz,
                       d_X, d_Y, d_Z,
                       wall1, wall2, useWalls,
                       threshDist,
                       FxL, FyL, FzL);

        // Calculate the velocity again with new forces. 
        VelocityUpdate(FxL[globalInd]/m, FyL[globalInd]/m, FzL[globalInd]/m,
                       0.5*delta_t,
                       vHalfX, vHalfY, vHalfZ, 
                       VxL+globalInd, VyL+globalInd, VzL+globalInd); 
        
                   
    
    }
}

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
                                          float threshDist){
    int cellInd = blockIdx.x;
    int nodeInd = threadIdx.x;
    int globalInd = cellInd*192+nodeInd;

    if (cellInd < No_of_C180s && nodeInd < 180){
        CalculateForce(nodeInd, cellInd, m,
                       internal_damping, delta_t, d_pressList,
                       Youngs_mod, R0, attraction_strength,
                       attraction_range, repulsion_strength,
                       repulsion_range, viscotic_damping,
                       gamma_visc, DL,
                       Minx, Miny, Minz,
                       xDiv, yDiv, zDiv,
                       VxL, VyL, VzL,
                       d_C180_nn, d_C180_sign, d_NoofNNlist,
                       d_NNlist, d_bounding_xyz,
                       d_X, d_Y, d_Z,
                       wall1, wall2, useWalls,
                       threshDist,
                       FxL, FyL, FzL);
    }
}

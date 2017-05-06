#include <stdio.h>
cudaDeviceProp getDevice(void)
{

  int deviceCount;
  cudaGetDeviceCount(&deviceCount);
  int device;
  cudaDeviceProp deviceProp;
  for (device = 0; device < deviceCount; ++device) {
       cudaGetDeviceProperties(&deviceProp, device);
       printf("   Device %s\n", deviceProp.name);
       printf("      compute capability           =         %d.%d\n", deviceProp.major, deviceProp.minor);
       printf("      totalGlobalMemory            =        %.2lf GB\n", deviceProp.totalGlobalMem/1000000000.0);
       printf("      l2CacheSize                  =    %8d\n", deviceProp.l2CacheSize);
       printf("      regsPerBlock                 =    %8d\n", deviceProp.regsPerBlock);
       printf("      multiProcessorCount          =    %8d\n", deviceProp.multiProcessorCount);
       printf("      maxThreadsPerMultiprocessor  =    %8d\n", deviceProp.maxThreadsPerMultiProcessor);
       printf("      sharedMemPerBlock            =    %8d B\n", (int)deviceProp.sharedMemPerBlock);
       printf("      warpSize                     =    %8d\n", deviceProp.warpSize);
       printf("      clockRate                    =    %8.2lf MHz\n", deviceProp.clockRate/1000.0);
       printf("      maxThreadsPerBlock           =    %8d\n", deviceProp.maxThreadsPerBlock);
       printf("      asyncEngineCount             =    %8d\n", deviceProp.asyncEngineCount);
       printf("      concurrentKernels            =    ");
       if(deviceProp.concurrentKernels==1) printf("   yes\n"); else printf("   no\n");

       printf("      ComputeMode                  =    %8d\n", deviceProp.computeMode);
       }

  return deviceProp; 
}



__device__ void CalcAndUpdateDaughtPos(int daughtInd, int partInd, float halfGap,
                                       float CMx, float CMy, float CMz,
                                       float X, float Y, float Z,
                                       float* d_XP,  float* d_YP,  float* d_ZP,
                                       float* d_X,  float* d_Y,  float* d_Z,
                                       float* d_XM,  float* d_YM,  float* d_ZM,
                                       float planeNx, float planeNy, float planeNz){
             
    // redefine position of parent cell wrt to an origin that includes
    // 0.5 the minimum gap between daughter cells
    X = X - CMx - halfGap*planeNx; 
    Y = Y - CMy - halfGap*planeNy; 
    Z = Z - CMz - halfGap*planeNz;

    float posDotN = X*planeNx + Y*planeNy + Z*planeNz;

    // If particle is below the plane, project onto the plane
    if (posDotN < 0.0f || posDotN == 0){
        X = X - posDotN*planeNx;
        Y = Y - posDotN*planeNy;
        Z = Z - posDotN*planeNz;
    }
             
    d_XP[daughtInd*192+partInd] = X + (CMx + halfGap*planeNx);
    d_YP[daughtInd*192+partInd] = Y + (CMy + halfGap*planeNy);
    d_ZP[daughtInd*192+partInd] = Z + (CMz + halfGap*planeNz);
    
    d_X[daughtInd*192+partInd] = X + (CMx + halfGap*planeNx);
    d_Y[daughtInd*192+partInd] = Y + (CMy + halfGap*planeNy);
    d_Z[daughtInd*192+partInd] = Z + (CMz + halfGap*planeNz);
    
    d_XM[daughtInd*192+partInd] = X + (CMx + halfGap*planeNx);
    d_YM[daughtInd*192+partInd] = Y + (CMy + halfGap*planeNy);
    d_ZM[daughtInd*192+partInd] = Z + (CMz + halfGap*planeNz);
}


__global__ void  cell_division(int rank,
                               float* d_XP, float* d_YP, float* d_ZP, 
                               float *d_X,  float *d_Y,  float *d_Z,
                               float* d_XM, float* d_YM, float* d_ZM,
                               float* AllCMx, float* AllCMy, float* AllCMz,
                               float* d_velListX, float* d_velListY, float* d_velListZ, 
                               int No_of_C180s, float *d_randNorm, float repulsion_range){
    int newrank = No_of_C180s;
    __shared__ float CMx, CMy, CMz;
  
    int tid  = threadIdx.x;
    int atom = tid;

    if (tid == 0){
        CMx = AllCMx[rank];
        CMy = AllCMy[rank];
        CMz = AllCMz[rank];
    }

    __syncthreads();


    if ( atom < 180 ) 
    {

        // planeN is the division plane's normal vector
        float planeNx = d_randNorm[0];
        float planeNy = d_randNorm[1];
        float planeNz = d_randNorm[2];

        if (abs(sqrt(planeNx*planeNx + planeNy*planeNy + planeNz*planeNz) - 1) > 1e-3){
            printf("OH SHIT: normal is not normalized\n");
            printf("Crash now :(\n"); 
            asm("trap;");
        }


        // First generate and write positions for the first daughter

        float X = d_X[rank*192+atom]; 
        float Y = d_Y[rank*192+atom]; 
        float Z = d_Z[rank*192+atom]; 
         
        CalcAndUpdateDaughtPos(rank, atom, 0.5*repulsion_range,
                               CMx, CMy, CMz,
                               X, Y, Z,
                               d_XP, d_YP, d_ZP, 
                               d_X, d_Y, d_Z,
                               d_XM, d_YM, d_ZM, 
                               planeNx, planeNy, planeNz);

        
        // Invert the normal
        planeNx = -1*planeNx; 
        planeNy = -1*planeNy; 
        planeNz = -1*planeNz;

        // Now repeat for the second daughter
        CalcAndUpdateDaughtPos(newrank, atom, 0.5*repulsion_range,
                               CMx, CMy, CMz,
                               X, Y, Z,
                               d_XP, d_YP, d_ZP, 
                               d_X, d_Y, d_Z,
                               d_XM, d_YM, d_ZM,
                               planeNx, planeNy, planeNz);

        // give the daughter the same velocities as the parent
        d_velListX[newrank*192 + atom] = d_velListX[rank*192+atom];
        d_velListY[newrank*192 + atom] = d_velListY[rank*192+atom];
        d_velListZ[newrank*192 + atom] = d_velListZ[rank*192+atom];
    }
}



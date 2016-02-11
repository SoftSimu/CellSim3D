#include <cuda.h>
#include <stdio.h>
#include "VectorFunctions.hpp"

__global__ void volumes( int No_of_C180s, int *C180_56,
                         float *X,    float *Y,   float *Z,
                         float *CMx , float *CMy, float *CMz, float *vol,
                         char* cell_div, float divVol, bool checkSphericity,
                         float* areaList){
    __shared__ float locX[192];
    __shared__ float locY[192];
    __shared__ float locZ[192];
    __shared__ float volume;
    __shared__ float volume2;
    __shared__ float area; 

    int fullerene = blockIdx.x;
    int tid       = threadIdx.x;

    if ( tid < 180 ){
        locX[tid] = X[192*fullerene+tid] -CMx[fullerene];
        locY[tid] = Y[192*fullerene+tid] -CMy[fullerene];
        locZ[tid] = Z[192*fullerene+tid] -CMz[fullerene];
    }

    if ( tid == 0){
        volume = 0.0f;
        volume2 = 0.0f; 
        area = 0.0f;
    }

    __syncthreads();

    if ( tid < 92 ){

        float avX = 0.0f;
        float avY = 0.0f;
        float avZ = 0.0f;

        for ( int i = 0; i < 5; ++i ){
            avX += locX[C180_56[7*tid+i]];
            avY += locY[C180_56[7*tid+i]];
            avZ += locZ[C180_56[7*tid+i]];
        }

        float avefactor = 0.166666667f;
        if ( tid < 12 )
        {
            avefactor = 0.2f;
        }
        else
        {
            avX += locX[C180_56[7*tid+5]];
            avY += locY[C180_56[7*tid+5]];
            avZ += locZ[C180_56[7*tid+5]];
        }

        avX *= avefactor;
        avY *= avefactor;
        avZ *= avefactor;

        float totvol = 0.0f;
        float totvol2 = 0.0f;
        float n1 = 0.0f;
        float n2 = 0.0f;
        float n3 = 0.0f;
        float faceArea = 0.0f;

        float face[7*3];

        float3 p0 = make_float3(avX, avY, avZ); 
        float3 p1, p2;

        for ( int i = 0; i < 6; ++i ){
            n1 = (locY[C180_56[7*tid+i+1]]*avZ-avY*locZ[C180_56[7*tid+i+1]])*locX[C180_56[7*tid+i]];
            n2 = (locZ[C180_56[7*tid+i+1]]*avX-avZ*locX[C180_56[7*tid+i+1]])*locY[C180_56[7*tid+i]];
            n3 = (locX[C180_56[7*tid+i+1]]*avY-avX*locY[C180_56[7*tid+i+1]])*locZ[C180_56[7*tid+i]];
            totvol += fabsf(n1+n2+n3);


            p1.x = locX[C180_56[7*tid+i]];
            p1.y = locY[C180_56[7*tid+i]];
            p1.z = locZ[C180_56[7*tid+i]];

            p2.x = locX[C180_56[7*tid+i+1]];
            p2.y = locY[C180_56[7*tid+i+1]];
            p2.z = locZ[C180_56[7*tid+i+1]];

            //totvol2 += dot(p0, cross(p1, p2)); 

            if (checkSphericity){
       
                // Get vectors that define a triangle 1, 2
                float x1 = locX[C180_56[7*tid+i]] - avX;
                float y1 = locY[C180_56[7*tid+i]] - avY;
                float z1 = locZ[C180_56[7*tid+i]] - avZ;

                float x2 = locX[C180_56[7*tid+i+1]] - avX;
                float y2 = locY[C180_56[7*tid+i+1]] - avY;
                float z2 = locZ[C180_56[7*tid+i+1]] - avZ;

                p1.x = p1.x - p0.x;
                p1.y = p1.y - p0.y;
                p1.z = p1.z - p0.z;

                p2.x = p2.x - p0.x;
                p2.y = p2.y - p0.y;
                p2.z = p2.z - p0.z;

                // now 1 will hold 1X2
                float xx = y1*z2 - z1*y2;
                float yy = z1*x2 - x1*z2;
                float zz = x1*y2 - y1*x2;

                // area of triangle is then 0.5*|1|
                faceArea += 0.5 * sqrt(xx*xx + yy*yy + zz*zz);
            }
        }
        atomicAdd(&volume, totvol);
        atomicAdd(&volume2, totvol2); 
    
        if (checkSphericity)
            atomicAdd(&area, faceArea); 
    }

    __syncthreads();

    if ( tid == 0){
        // if (isnan(vol[fullerene])){
        //     printf("OH SHIT: nan volume for cell %d", fullerene); 
        //     asm("trap;");
        // }

        
     
        volume = volume/6.0; 
        vol[fullerene] = volume;
        bool divide = 0; 
     
        if (volume > divVol){
            cell_div[fullerene] = 1;
            //printf("Cell %d volume =%f, volume2=%f \n", fullerene, volume/6.0, volume2/6.0); 
            //divide = 1; 
        }

        if (checkSphericity){
            areaList[fullerene] = area; 
            float psi = 4.835975862049408 * powf(volume, 2.0/3.0)/area;
            if ((1.0f - psi) > 0.1){
                cell_div[fullerene] = 0;
                //printf("cell %d division rejected\n", fullerene);
            }
        }
    }
    
}


__global__ void NewVolumes( int No_of_C180s, int *C180_56,
                            float *X,    float *Y,   float *Z,
                            float *CMx , float *CMy, float *CMz, float *voll,
                            char* cell_div, float divVol)
{
    __shared__ float locX[180];
    __shared__ float locY[180];
    __shared__ float locZ[180];
    __shared__ float volume;

    int blockInd = blockIdx.x;
    int cellInd = threadIdx.x;

    for (int i = 0; i < 180; i++){
        locX[i] = X[cellInd*192+i] - CMx[cellInd];
        locY[i] = Y[cellInd*192+i] - CMy[cellInd];
        locZ[i] = Z[cellInd*192+i] - CMz[cellInd];
    }
    
    float vol = 0.0f;
    float area = 0.0f;
    int N1, N2, N3, N4, N5, N6, N7;
    float CMFx, CMFy, CMFz;
    float b1x, b1y, b1z;
    float b2x, b2y, b2z;

    float volFace = 0.0f;
    float CPx, CPy, CPz; 
    
    for (int face = 0; face < 92; face++){
        N1 = C180_56[7*face+1];
        N2 = C180_56[7*face+2];
        N3 = C180_56[7*face+3];
        N4 = C180_56[7*face+4];
        N5 = C180_56[7*face+5];
        N6 = C180_56[7*face+6];
        
        CMFx = 1/6.0 * (locX[N1] + locX[N2] + locX[N3] +
                        locX[N4] + locX[N5] + locX[N6]);
        
        CMFy = 1/6.0 * (locY[N1] + locY[N2] + locY[N3] +
                        locY[N4] + locY[N5] + locY[N6]);
        
        CMFz = 1/6.0 * (locZ[N1] + locZ[N2] + locZ[N3] +
                        locZ[N4] + locZ[N5] + locZ[N6]);

        // each face makes a hex-prism (extruded hexagon)
        // the volume of which can be estimated by volume of 3
        // parallelipipeds
        // volume defined as u.(v x w) where u, v, w define length, width
        // height vectors of the piped.

        b1x = locX[N1] - locX[N2]; 
        b1y = locY[N1] - locY[N2]; 
        b1z = locZ[N1] - locZ[N2]; 

        b2x = locX[N3] - locX[N2];
        b2y = locY[N3] - locY[N2];
        b2z = locZ[N3] - locZ[N2];

        CPx = b1y*b2x - b1z*b2y;
        CPy = b1x*b2z - b1z*b2x;
        CPz = b1x*b2y - b1y*b2x;

        area += sqrt(CPx*CPx + CPy*CPy + CPz*CPz); 
        
        volFace += fabs(CMFx*CPx - CMFy*CPy + CMFz*CPz);
        
        
        b1x = locX[N3] - locX[N4]; 
        b1y = locY[N3] - locY[N4]; 
        b1z = locZ[N3] - locZ[N4]; 

        b2x = locX[N5] - locX[N4];
        b2y = locY[N5] - locY[N4];
        b2z = locZ[N5] - locZ[N4];
        
        CPx = b1y*b2x - b1z*b2y;
        CPy = b1x*b2z - b1z*b2x;
        CPz = b1x*b2y - b1y*b2x;

        area += sqrt(CPx*CPx + CPy*CPy + CPz*CPz); 
        
        volFace += fabs(CMFx*CPx - CMFy*CPy + CMFz*CPz);        
        

        b1x = locX[N5] - locX[N6]; 
        b1y = locY[N5] - locY[N6]; 
        b1z = locZ[N5] - locZ[N6]; 
        
        b2x = locX[N1] - locX[N6];
        b2y = locY[N1] - locY[N6];
        b2z = locZ[N1] - locZ[N6];
        
        CPx = b1y*b2x - b1z*b2y;
        CPy = b1x*b2z - b1z*b2x;
        CPz = b1x*b2y - b1y*b2x;

        area += sqrt(CPx*CPx + CPy*CPy + CPz*CPz); 
        
        volFace += fabs(CMFx*CPx - CMFy*CPy + CMFz*CPz);
        
        vol += volFace;
    }

    
}

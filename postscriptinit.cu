
#include <stdio.h>
#include <math.h>

int  PSSETUP (FILE *outfile)
   {
   fprintf(outfile,"%%!PS-Adobe-2.0\n");
   fprintf(outfile,"50 550 translate\n");
   fprintf(outfile,"7.0 7.0 scale\n");
   fprintf(outfile,"0.015 setlinewidth\n");
   fprintf(outfile,"/circledict 8 dict def\n");
   fprintf(outfile,"circledict /mtrx matrix put\n");
   fprintf(outfile,"/circle { circledict begin\n");
   fprintf(outfile,"  /rad exch def\n");
   fprintf(outfile,"  /y exch def\n");
   fprintf(outfile,"  /x exch def\n");
   fprintf(outfile,"  /savematrix mtrx currentmatrix def\n");
   fprintf(outfile,"  x y translate\n");
   fprintf(outfile,"  rad rad scale\n");
   fprintf(outfile,"  0 0 1 0 360 arc\n");
   fprintf(outfile,"  savematrix setmatrix\n");
   fprintf(outfile,"  end } def\n");
   fprintf(outfile,"/Times-Roman findfont 0.3 scalefont setfont\n");
   fprintf(outfile,"/Times-Roman findfont 0.15 scalefont setfont\n");
   return(0);
   }



int PSLINE(float X1, float Y1, float X2, float Y2, FILE *outfile)
   {
//C      Prints PS-code drawing a line from (X1,Y1) to (X2,Y2) to file PSUNIT

   fprintf(outfile,"%7.2f %7.2f moveto\n",X1,Y1);
   fprintf(outfile,"%7.2f %7.2f lineto\n",X2,Y2);
   fprintf(outfile,"stroke\n");

   return(0);
   }




int PSCIRCLE(float X,float Y,FILE *outfile)
   {
   float RADIUS;

   RADIUS=0.57/2.0;

   fprintf(outfile,"0 0 0 setrgbcolor");
   fprintf(outfile,"newpath");
   fprintf(outfile,"%12.2f %12.2f %12.2f circle\n", X, Y, RADIUS);
   fprintf(outfile,"closepath");
   fprintf(outfile,"stroke");

   return(0);
   }




int PSNUM(float X, float Y, int NUMBER, FILE *outfile)
   {
   fprintf(outfile,"%7.2f %7.2f moveto\n",X, Y );
   fprintf(outfile," (%d) show\n", NUMBER);

   return(0);
   }

int PSNET(int NN,int sl,float L1, float *X, float *Y, float *Z, int CCI[2][271])
  {
  int j,rank;
  float X1,Y1,Z1,X2,Y2,Z2;
  float SX1, SX2, SZ1, SZ2;
  FILE *outfile;

  outfile = fopen("psfil","a");
  if ( outfile == NULL ) {printf("Unable to open file psfil\n");return(-1);}

  PSSETUP(outfile);

  for ( rank = 0; rank < NN/270; ++rank )
      for ( j = 0; j < 270; ++j )
         {
         X1 = X[192*rank+CCI[0][j]];
         X2 = X[192*rank+CCI[1][j]];
         Y1 = Y[192*rank+CCI[0][j]];
         Y2 = Y[192*rank+CCI[1][j]];
         Z1 = Z[192*rank+CCI[0][j]];
         Z2 = Z[192*rank+CCI[1][j]];
         PSLINE(X1,Y1,X2,Y2,outfile);
         SX1=X1+40.0;
         SX2=X2+40.0;
         SZ1=Z1+10.0;
         SZ2=Z2+10.0;
         PSLINE(SX1,SZ1,SX2,SZ2,outfile);
         SX1=X1+40.0;
         SX2=X2+40.0;
         SZ1=Z1-10.0;
         SZ2=Z2-10.0;
         if  (fabsf((Y1+Y2)/2.0-sl*L1) < 0.2) PSLINE(SX1,SZ1,SX2,SZ2,outfile);
         fprintf(outfile,"stroke\n");
         }

  fprintf(outfile,"showpage\n");

  fclose(outfile);
  return(0);
  }


//C *****************************************************************



int getDevice(void)
{

  int deviceCount;
  cudaGetDeviceCount(&deviceCount);
  int device;
  for (device = 0; device < deviceCount; ++device) {
       cudaDeviceProp deviceProp;
       cudaGetDeviceProperties(&deviceProp, device);
       printf("   Device %s\n", deviceProp.name);
       printf("      compute capability           =         %d.%d\n", deviceProp.major, deviceProp.minor);
       printf("      totalGlobalMemory            =        %.2lf GB\n", deviceProp.totalGlobalMem/1000000000.0);
//       printf("      l2CacheSize                  =    %8d\n", deviceProp.l2CacheSize);
       printf("      regsPerBlock                 =    %8d\n", deviceProp.regsPerBlock);
       printf("      multiProcessorCount          =    %8d\n", deviceProp.multiProcessorCount);
       printf("      maxThreadsPerMultiprocessor  =    %8d\n", deviceProp.maxThreadsPerMultiProcessor);
       printf("      sharedMemPerBlock            =    %8d B\n", (int)deviceProp.sharedMemPerBlock);
//       printf("      warpSize                     =    %8d\n", deviceProp.warpSize);
       printf("      clockRate                    =    %8.2lf MHz\n", deviceProp.clockRate/1000.0);
//       printf("      maxThreadsPerBlock           =    %8d\n", deviceProp.maxThreadsPerBlock);
//       printf("      asyncEngineCount             =    %8d\n", deviceProp.asyncEngineCount);
//       printf("      concurrentKernels            =    ");
//       if(deviceProp.concurrentKernels==1) printf("   yes\n"); else printf("   no\n");
       }

return(0);
}


__global__ void  cell_division(int rank, float *d_XP, float *d_YP, float *d_ZP,
                                         float *d_X,  float *d_Y,  float *d_Z,
                               int No_of_C180s, float *d_ran2, float repulsion_range)
  {
__shared__ float  sumx[256];
__shared__ float  sumy[256];
__shared__ float  sumz[256];
  int newrank;
  float RCNNS, SCNNS, TCNNS;
__shared__ float CMx, CMy, CMz;
  
int tid  = threadIdx.x;
int atom = tid;

sumx[tid] = 0.0;
sumy[tid] = 0.0;
sumz[tid] = 0.0;

if ( tid < 180 )
    {
    sumx[tid] = d_XP[rank*192+tid];
    sumy[tid] = d_YP[rank*192+tid];
    sumz[tid] = d_ZP[rank*192+tid];
    }

__syncthreads();

for ( int s = blockDim.x/2; s > 0; s>>=1)
   {
   if ( tid < s )
      {
      sumx[tid] += sumx[tid+s];
      sumy[tid] += sumy[tid+s];
      sumz[tid] += sumz[tid+s];
      }
   __syncthreads();
   }

if ( tid == 0 ) 
   {
   CMx = sumx[0]/180.0f;
   CMy = sumy[0]/180.0f;
   CMz = sumz[0]/180.0f;
   }

__syncthreads();


  if ( atom < 180 ) 
     {

     float ALP =d_ran2[1];
     float ALP2=d_ran2[0];

     float SC =  cosf(ALP)*cosf(ALP2)*(d_XP[192*rank+atom]-CMx)
                -sinf(ALP)*cosf(ALP2)*(d_YP[rank*192+atom]-CMy)
                +sinf(ALP2)*(d_ZP[rank*192+atom]-CMz);
     float TC =  sinf(ALP)*(d_XP[rank*192+atom]-CMx)
                +cosf(ALP)*(d_YP[rank*192+atom]-CMy);
     float RC = -sinf(ALP2)*cosf(ALP)*(d_XP[rank*192+atom]-CMx)
                +sinf(ALP)*sinf(ALP2)*(d_YP[rank*192+atom]-CMy)
                +cosf(ALP2)*(d_ZP[rank*192+atom]-CMz);


     newrank = No_of_C180s;                
  
     SCNNS = SC;
     TCNNS = TC;
     if ( RC > 0.0f )
         {
         RCNNS = -repulsion_range/2.0f;
         if ( RC < repulsion_range/2.0f ) RC = repulsion_range/2.0f;
         }
     else
         {
         RCNNS = RC;
         if ( RCNNS > -repulsion_range/2.0f ) RCNNS = -repulsion_range/2.0f;
         RC = repulsion_range/2.0f;
         }


     d_XP[192*rank+atom]    =  cosf(-ALP)*cosf(-ALP2)*SC -sinf(-ALP)*TC + cosf(-ALP)*sinf(-ALP2)*RC+CMx;
     d_YP[192*rank+atom]    =  sinf(-ALP)*cosf(-ALP2)*SC +cosf(-ALP)*TC + sinf(-ALP)*sinf(-ALP2)*RC+CMy;
     d_ZP[192*rank+atom]    =            -sinf(-ALP2)*SC                            +cosf(-ALP2)*RC+CMz;

     d_XP[192*newrank+atom] =  cosf(-ALP)*cosf(-ALP2)*SCNNS-sinf(-ALP)*TCNNS + cosf(-ALP)*sinf(-ALP2)*RCNNS+CMx;
     d_YP[192*newrank+atom] =  sinf(-ALP)*cosf(-ALP2)*SCNNS+cosf(-ALP)*TCNNS + sinf(-ALP)*sinf(-ALP2)*RCNNS+CMy;
     d_ZP[192*newrank+atom] =            -sinf(-ALP2)*SCNNS                  +            cosf(-ALP2)*RCNNS+CMz;

        
     d_X[rank*192+atom ]   = d_XP[rank*192+atom];
     d_Y[rank*192+atom ]   = d_YP[rank*192+atom];
     d_Z[rank*192+atom ]   = d_ZP[rank*192+atom];

     d_X[newrank*192+atom] = d_XP[newrank*192+atom];
     d_Y[newrank*192+atom] = d_YP[newrank*192+atom];
     d_Z[newrank*192+atom] = d_ZP[newrank*192+atom];
     }


}



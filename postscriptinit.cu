
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

  return deviceProp; 
}


__global__ void  cell_division(int rank,
                               float *d_XP, float *d_YP, float *d_ZP,
                               float *d_X,  float *d_Y,  float *d_Z,
                               float* AllCMx, float* AllCMy, float* AllCMz, 
                               int No_of_C180s, float *d_ran2, float repulsion_range)
  {
  int newrank;
  float RCNNS, SCNNS, TCNNS;
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
         float planeNx = d_ran2[2];
         float planeNy = d_ran2[3];
         float planeNz = d_ran2[4];

     
         if (abs(sqrt(planeNx*planeNx + planeNy*planeNy + planeNz*planeNz) - 1) > 1e-3){
             printf("wtf\n");
             asm("trap;");
         }

         float X = d_XP[rank*192+atom] - CMx; 
         float Y = d_YP[rank*192+atom] - CMy; 
         float Z = d_ZP[rank*192+atom] - CMz;

         float posDotN = X*planeNx + Y*planeNy + Z*planeNz;

     
         // d1 is the position of atom belonging to first daughter
         float d1X = 0;
         float d1Y = 0;
         float d1Z = 0;
     
         // d2 is the position of atom belonging to second daughter
         float d2X = 0;
         float d2Y = 0;
         float d2Z = 0;


     
         // If atom is below the division plane or in the division plane
         if (posDotN < 0 || posDotN == 0){
         
             // Project it onto the plane, and move it ABOVE the plane by
             // 0.5*repulsion_range, give its new position to FIRST daughter
             d1X = X - posDotN*planeNx + 0.5*repulsion_range*planeNx;
             d1Y = Y - posDotN*planeNy + 0.5*repulsion_range*planeNy;
             d1Z = Z - posDotN*planeNz + 0.5*repulsion_range*planeNz;
         
             // Give atom's original position to SECOND daughter unchanged. 
             d2X = X;
             d2Y = Y;
             d2Z = Z;
         
         } else { // if atom is above the division plane
         
             // Give atom's original position to FIRST daughter unchanged
             d1X = X;
             d1Y = Y;
             d1Z = Z;

             // Project it onto the plane, and move it UNDER the plane by
             // 0.5*repulsion_range, give its new position to SECOND daughter
             d2X = X - posDotN*planeNx - 0.5*repulsion_range*planeNx;
             d2Y = Y - posDotN*planeNy - 0.5*repulsion_range*planeNy;
             d2Z = Z - posDotN*planeNz - 0.5*repulsion_range*planeNz;
         
         }

     
         d_XP[rank*192+atom] = d1X + CMx;
         d_YP[rank*192+atom] = d1Y + CMy;
         d_ZP[rank*192+atom] = d1Z + CMz;

         d_XP[newrank*192+atom] = d2X + CMx;
         d_YP[newrank*192+atom] = d2Y + CMy;
         d_ZP[newrank*192+atom] = d2Z + CMz;

         d_X[rank*192+atom ]   = d_XP[rank*192+atom];
         d_Y[rank*192+atom ]   = d_YP[rank*192+atom];
         d_Z[rank*192+atom ]   = d_ZP[rank*192+atom];

         d_X[newrank*192+atom] = d_XP[newrank*192+atom];
         d_Y[newrank*192+atom] = d_YP[newrank*192+atom];
         d_Z[newrank*192+atom] = d_ZP[newrank*192+atom];
     
     }

}



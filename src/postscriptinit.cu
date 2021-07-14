
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



cudaDeviceProp getDevice( int idev)
{

  //int deviceCount;
  //cudaGetDeviceCount(&deviceCount);
  
  int device;
  cudaDeviceProp deviceProp;
  
  device = idev;
  //for (device = 0; device < deviceCount; ++device) {
  cudaGetDeviceProperties(&deviceProp, device);
  printf("   Device %s\n", deviceProp.name);
  printf("      compute capability           =    %d.%d\n", deviceProp.major, deviceProp.minor);
  printf("      totalGlobalMemory            =    %.2lf GB\n", deviceProp.totalGlobalMem/1000000000.0);
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
  //}

  return deviceProp; 
}

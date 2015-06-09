#include <netcdf.h>
#include "BinaryOutput.h"
#include <stdio.h>

#define NETCDF_ERR_CODE 13
#define NETCDF_ERR(e) {printf("Error: %s\n", nc_strerror(e)); exit (NETCDF_ERR_CODE);}

int OpenBinaryFile(char* FileName){
    int fileId, retVal;
    if ((retVal = nc_create(FileName, NC_CLOBBER, &fileId)))
        NETCDF_ERR(retVal);
    
    return fileId; 
}

int WriteToFile(float* posX, float* posY, float* posZ,
                // Add the variables below iff needed
                //float* velX, float* velY, float* velZ,
                //float* CoMX, float* CoMY, float* CoMZ,
                int timeStep, int fileId){
    printf("Testing..\n");
    return 0;
}


void CloseBinaryFile(int fileId){
    int retVal;
    if ((retVal = nc_close(fileId)))
        NETCDF_ERR(retVal);
    
    return;
}

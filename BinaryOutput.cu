#include <netcdf.h>
#include "BinaryOutput.h"
#include <stdio.h>

inline void NetcdfErrCheck(int funcReturn){
    // all netcdf functions return 0 if they succeed
    if (funcReturn != 0){
        printf("NetCDF error!\n");
        printf("Error: %s\n", nc_strerror(funcReturn));
        printf("In file %s, at line %d\n", __FILE__, __LINE__); 
        exit (NETCDF_ERR_CODE); 
    }
        
}


void OpenBinaryFile(char* fileName, BinFileAttrStruct* bFA){
    NetcdfErrCheck(nc_create(fileName, NC_NETCDF4|NC_CLOBBER , &(bFA->fileId)));
    
    // Now setup up the parameters that need to be written

    NetcdfErrCheck(nc_def_dim(bFA->fileId,
                              "TimeStep", NC_UNLIMITED, &(bFA->timeId)));
    NetcdfErrCheck(nc_def_dim(bFA->fileId,
                              "NodeIndex", NC_UNLIMITED, &(bFA->indId)));

    
    int dimVec[2] = {bFA->timeId, bFA->indId}; 
    
    NetcdfErrCheck(nc_def_var(bFA->fileId,
                              "XPosition", NC_FLOAT, 2, dimVec,
                              &(bFA->XId)));
     NetcdfErrCheck(nc_def_var(bFA->fileId,
                               "YPosition", NC_FLOAT, 2, dimVec,
                               &(bFA->YId)));
     NetcdfErrCheck(nc_def_var(bFA->fileId,
                               "ZPosition", NC_FLOAT, 2, dimVec,
                               &(bFA->ZId)));
        
    
}

void WriteToBinaryFile(float* posX, float* posY, float* posZ,
                       int numCells, int timeStep, BinFileAttrStruct* bFA){

    printf("Testing the binary data output...\n");

    const size_t start[2] = {timeStep/1000, 0};
    const size_t count[2] = {1, numCells*192};
    NetcdfErrCheck(nc_put_vara_float(bFA->fileId, bFA->XId, start, count,
                                     &posX[0]));
     NetcdfErrCheck(nc_put_vara_float(bFA->fileId, bFA->YId, start, count,
                                      posY));
     NetcdfErrCheck(nc_put_vara_float(bFA->fileId, bFA->ZId, start, count,
                                      posZ)); 
}


void CloseBinaryFile(BinFileAttrStruct* bFA){
    NetcdfErrCheck(nc_close(bFA->fileId));
}

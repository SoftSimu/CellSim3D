#include <netcdf.h>
#define NETCDF_ERR_CODE 13

inline void NetcdfErrCheck(int funcReturn);

// Set up data structure to hold the id information
// of all our data
struct BinFileAttrStruct{
    int fileId;
    int XId, YId, ZId;
    int timeId, indId, scalDimId;
    int trajWriteIntId;
};


// Opens and sets up netcdf file
void OpenBinaryFile(char* FileName, BinFileAttrStruct* bFA,
                    int trajWriteInt);
// Writes data at time = timeStep
void WriteToBinaryFile(float* posX, float* posY, float* posZ,
                      // Add the variables below iff needed
                      //float* velX, float* velY, float* velZ,
                      //float* CoMX, float* CoMY, float* CoMZ,
                      int timeStep, int fileId, BinFileAttrStruct* bFA);

// Closes netcdf file with id fileId
void CloseBinaryFile(BinFileAttrStruct* bFA); // input file handle

// Returns a netcdf file ids which is just an int
int OpenBinaryFile(char* FileName);


// Writes data at time = timeStep
int WriteToBinaryFile(float* posX, float* posY, float* posZ,
                      // Add the variables below iff needed
                      //float* velX, float* velY, float* velZ,
                      //float* CoMX, float* CoMY, float* CoMZ,
                      int timeStep, int fileId);

// Closes netcdf file with id fileId
void CloseBinaryFile(int fileId); // input file handle

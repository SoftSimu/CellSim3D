// This file contains some general purpose device only functions

#ifndef DEVICE_FUNCTIONS_CUH
#define DEVICE_FUNCTIONS_CUH

__host__ __device__ float dot(float3 a, float3 b);

__host__ __device__ float3 cross(float3 a, float3 b);

__host__ __device__ float mag2(float3 a);

__host__ __device__ float mag(float3 a);

#endif // DEVICE_FUNCTIONS_CUH

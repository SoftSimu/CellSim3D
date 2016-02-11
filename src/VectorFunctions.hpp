// This file contains some general purpose device only functions

#ifndef DEVICE_FUNCTIONS_CUH
#define DEVICE_FUNCTIONS_CUH

__host__ __device__ inline float dot(float3 a, float3 b){
    return a.x*a.x + a.y*a.y + a.z*a.z;
}


__host__ __device__ inline float3 cross(float3 a, float3 b){
    float3 c;

    c.x = a.y*b.z - a.z*b.y;
    c.y = a.z*b.x - a.x*b.z;
    c.z = a.x*b.y - a.y*b.x;

    return c;
}

__host__ __device__ inline float mag2(float3 a){
    return dot(a,a);
}

__host__ __device__ inline float mag(float3 a){
    return sqrtf(mag2(a));
}


__host__ __device__ inline float3 operator+(float3 a, float3 b){
    a.x = a.x + b.x;
    a.y = a.y + b.y;
    a.z = a.z + b.z;
}

#endif // DEVICE_FUNCTIONS_CUH

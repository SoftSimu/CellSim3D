// This file contains some general purpose device only functions

#ifndef VECTOR_FUNCTIONS_HPP
#define VECTOR_FUNCTIONS_HPP
#include <stdio.h>
#include <thrust/device_vector.h>
#include <thrust/fill.h>
#define MAX_NN 1024

struct angles3{
    float aij, ajk, aik;
};

struct R3Nptrs{
    float* x;
    float* y;
    float* z;
};

typedef struct R3Nptrs R3Nptrs;

__host__ __device__ inline angles3 make_angles3(float aij, float ajk,
                                                float aik){
    angles3 A;
    A.aij = aij;
    A.ajk = ajk;
    A.aik = aik;
    return A;
}

__host__ __device__ inline float dot(float3 a, float3 b){
    return a.x*b.x + a.y*b.y + a.z*b.z;
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


__host__ __device__ inline float3 operator+(const float3 a, const float3 b){
    float3 c;
    c.x = a.x + b.x;
    c.y = a.y + b.y;
    c.z = a.z + b.z;
    return c;
}


__host__ __device__ inline float3 operator-(const float3 a, const float3 b){
    float3 c;
    c.x = a.x - b.x;
    c.y = a.y - b.y;
    c.z = a.z - b.z;
    return c;
}

__host__ __device__ inline float3 operator*(const float a, const float3 b){
    float3 c;
    c.x = a*b.x;
    c.y = a*b.y;
    c.z = a*b.z;
    return c;
}

__host__ __device__ inline float3 operator*(const float3 b, const float a){
    return a*b;
}

__host__ __device__ inline float3 operator/(const float3 b, const float a){
    // float3 c;
    // c.x = b.x/a;
    // c.y = b.y/a;
    // c.z = b.z/a;
    // return c;

    return (1/a)*b;
}

__host__ __device__ inline bool operator==(const float3 &a, const float3 &b){
    if (a.x == b.x &&
        a.y == b.y &&
        a.z == b.z)
        return true;

    return false;
}

__host__ __device__ inline bool operator!=(const float3 &a, const float3 &b){
    return !(a==b);
}

__host__ __device__ inline float3 calcUnitVec(const float3 a){
    return a/mag(a);
}

__host__ __device__ inline void print_float3(float3 a){
    printf("(%f, %f, %f)", a.x, a.y, a.z);
}

__host__ __device__ inline bool good_float3(float3 a){
    if ( !isfinite(a.x) ||
         !isfinite(a.y) ||
         !isfinite(a.z) )
        return false;
    return true;

}
#endif // VECTOR_FUNCTIONS_HPP

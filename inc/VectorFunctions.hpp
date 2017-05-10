// This file contains some general purpose device only functions
#ifndef VECTOR_FUNCTIONS_CUH
#define VECTOR_FUNCTIONS_CUH
#include <stdio.h>
#include <thrust/device_vector.h>
#include <thrust/fill.h>
#include "Types.cuh"

__host__ __device__ inline angles3 make_angles3(real aij, real ajk,
                                                real aik){
    angles3 A;
    A.aij = aij;
    A.ajk = ajk;
    A.aik = aik;
    return A;
}

__host__ __device__ inline real dot(real3 a, real3 b){
    return a.x*b.x + a.y*b.y + a.z*b.z;
}


__host__ __device__ inline real3 cross(real3 a, real3 b){
    real3 c;

    c.x = a.y*b.z - a.z*b.y;
    c.y = a.z*b.x - a.x*b.z;
    c.z = a.x*b.y - a.y*b.x;

    return c;
}

__host__ __device__ inline real mag2(real3 a){
    return dot(a,a);
}

__host__ __device__ inline real mag(real3 a){
    return sqrtf(mag2(a));
}


__host__ __device__ inline real3 operator+(const real3 a, const real3 b){
    real3 c;
    c.x = a.x + b.x;
    c.y = a.y + b.y;
    c.z = a.z + b.z;
    return c;
}


__host__ __device__ inline real3 operator-(const real3 a, const real3 b){
    real3 c;
    c.x = a.x - b.x;
    c.y = a.y - b.y;
    c.z = a.z - b.z;
    return c;
}

__host__ __device__ inline real3 operator*(const real a, const real3 b){
    real3 c;
    c.x = a*b.x;
    c.y = a*b.y;
    c.z = a*b.z;
    return c;
}

__host__ __device__ inline real3 operator*(const real3 b, const real a){
    return a*b;
}

__host__ __device__ inline real3 operator/(const real3 b, const real a){
    return (1/a)*b;
}

__host__ __device__ inline bool operator==(const real3 &a, const real3 &b){
    if (a.x == b.x &&
        a.y == b.y &&
        a.z == b.z)
        return true;

    return false;
}

__host__ __device__ inline bool operator!=(const real3 &a, const real3 &b){
    return !(a==b);
}

__host__ __device__ inline real3 calcUnitVec(const real3 a){
    return a/mag(a);
}

__host__ __device__ inline void print_real3(real3 a){
    printf("(%f, %f, %f)", a.x, a.y, a.z);
}

__host__ __device__ inline bool good_real3(real3 a){
    if ( !isfinite(a.x) ||
         !isfinite(a.y) ||
         !isfinite(a.z) )
        return false;
    return true;

}

real3 make_real3(real x, real y, real z){
    real3 a;
    a.x = x;
    a.y = y;
    a.z = z;
    return a;
}
#endif // VECTOR_FUNCTIONS_CUH

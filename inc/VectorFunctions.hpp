// This file contains some general purpose device only functions

#ifndef VECTOR_FUNCTIONS_HPP
#define VECTOR_FUNCTIONS_HPP
#include <stdio.h>
#include <thrust/device_vector.h>
#include <thrust/fill.h>
#define MAX_NN 256

struct angles3{
    float aij, ajk, aik;
};

struct R3NVectors{

    thrust::device_vector<float> x;
    thrust::device_vector<float> y;
    thrust::device_vector<float> z;
};

typedef struct R3NVectors R3NVectors;

struct hR3NVectors{

    thrust::host_vector<float> x;
    thrust::host_vector<float> y;
    thrust::host_vector<float> z;
};

typedef struct hR3NVectors hR3NVectors;

inline void InitR3NVecs(R3NVectors& a, int size){
    a.x.resize(size);
    a.y.resize(size);
    a.z.resize(size);

    thrust::fill(a.x.begin(), a.x.end(), 0.f);
    thrust::fill(a.y.begin(), a.y.end(), 0.f);
    thrust::fill(a.z.begin(), a.z.end(), 0.f);
}


inline void InitR3NVecs(hR3NVectors& a, int size){
    a.x.resize(size);
    a.y.resize(size);
    a.z.resize(size);

    thrust::fill(a.x.begin(), a.x.end(), 0.f);
    thrust::fill(a.y.begin(), a.y.end(), 0.f);
    thrust::fill(a.z.begin(), a.z.end(), 0.f);
}

inline void CopyR3NvecsToHost(hR3NVectors& dest, R3NVectors& source, int n){
    thrust::copy_n(source.x.begin(), n, dest.x.begin());
    thrust::copy_n(source.y.begin(), n, dest.y.begin());
    thrust::copy_n(source.z.begin(), n, dest.z.begin());
}

struct R3Nptrs{
    float* x;
    float* y;
    float* z;
};

typedef struct R3Nptrs R3Nptrs;

inline void setR3Nptrs (R3Nptrs& a, R3NVectors& b){
    a.x = thrust::raw_pointer_cast(&(b.x[0]));
    a.y = thrust::raw_pointer_cast(&(b.y[0]));
    a.z = thrust::raw_pointer_cast(&(b.z[0]));
}

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

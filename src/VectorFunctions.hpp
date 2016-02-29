// This file contains some general purpose device only functions

#ifndef VECTOR_FUNCTIONS_HPP
#define VECTOR_FUNCTIONS_HPP

struct angles3{
    float aij, ajk, aik;
};

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
    float3 c;
    c.x = a*b.x;
    c.y = a*b.y;
    c.z = a*b.z;
    return c;
}

__host__ __device__ inline float3 operator/(const float3 b, const float a){
    float3 c;
    c.x = b.x/a;
    c.y = b.y/a;
    c.z = b.z/a;
    return c;
}
#endif // VECTOR_FUNCTIONS_HPP

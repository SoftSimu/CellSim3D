#ifndef ADAPTIVE_TIME_KERNELS
#define ADAPTIVE_TIME_KERNELS

// __host__ __device__ float get_pn2(float c1, float c2);

// __host__ __device__ float get_pn1(float c1, float c2);

// __host__ __device__ float get_pn0(float c1, float c2);

// __host__ __device__ float get_pp1(float c1, float c2);

// __host__ __device__ float get_cn2(float c1, float c2);

// __host__ __device__ float get_cn1(float c1, float c2);

// __host__ __device__ float get_c0(float c1, float c2);

// __host__ __device__ float get_cp1(float c1, float c2);

// __host__ __device__ float get_alpha(float c1, float c2);

// __host__ __device__ float get_beta(float c1, float c2);

// predictor stuff
inline float get_pn2(float c1, float c2){
    return 2*(1-c1)/((c2-c1)*c2*(1+c2));
}

inline float get_pn1(float c1, float c2){
    return 2*(c2-1)/(c1*(1+c1)*(c2-c1));
}

inline float get_pn0(float c1, float c2){
    return -2*(c1+c2-1)/(c1*c2); 
}

inline float get_pp1(float c1, float c2){
    return 2*(c1+c2)/((1+c1)*(1+c2));
}

// corrector
inline float get_cn2(float c1, float c2){
    return -2*(2+c1)/((c2 - c1)*c2*(1 + c2));
}

inline float get_cn1(float c1, float c2){
    return 2*(2+c2)/(c1*(1+c1)*(c2-c1)); 
}

inline float get_cn0(float c1, float c2){
    return -2*(2+c1+c2)/(c1*c2); 
}

inline float get_cp1(float c1, float c2){ 
    return 2*(3+c1+c2)/((1+c1)*(1+c2)); 
}

inline float get_alpha(float c1, float c2){
    return -(c1*c2-c1-c2)/12;
}

inline float get_beta(float c1, float c2){
    return -(3+2*c1 + 2*c2 + c1*c2)/12;
}



struct adp_coeffs{
    float kn1, kn2, k0, k1;
};



inline adp_coeffs getAdpCoeffs(bool isPredictor, float c1, float c2){
    adp_coeffs a;
    if (isPredictor){
        a.kn1 = get_pn1(c1, c2);
        a.kn2 = get_pn2(c1, c2);
        a.k0 = get_pn0(c1, c2);
        a.k1 = get_pp1(c1, c2);
    } else {
        a.kn1 = get_cn1(c1, c2);
        a.kn2 = get_cn2(c1, c2);
        a.k0 = get_cn0(c1, c2);
        a.k1 = get_cp1(c1, c2);
    }

    return a; 
}

    
__global__ void Integrate(float *d_XP, float *d_YP, float *d_ZP,
                          float *d_X, float *d_Y, float *d_Z, 
                          float *d_XM, float *d_YM, float *d_ZM,
                          float *d_XMM, float *d_YMM, float *d_ZMM,
                          float *d_velListX, float *d_velListY, float *d_velListZ, 
                          float *d_time, float mass,
                          float3 *d_forceList, int numCells, adp_coeffs a);

__global__ void ForwardTime(float *d_XP, float *d_YP, float *d_ZP,
                            float *d_X, float *d_Y, float *d_Z,
                            float *d_XM, float *d_YM, float *d_ZM, 
                            float *d_XMM, float *d_YMM, float *d_ZMM,
                            int numCells);

__global__ void ComputeTimeUpdate(float *d_XP, float *d_YP, float *d_ZP, 
                                  float *d_Xt, float *d_Yt, float *d_Zt,
                                  float *d_AdpErrors, float *d_time, float dt_max,
                                  float alpha, float beta, int numCells);

#endif // ADAPTIVE_TIME_KERNELS

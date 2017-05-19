#ifndef TYPES_CUH
#define TYPES_CUH
#define REAL_TYPE_F32
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/copy.h>
#include <vector>

#ifdef REAL_TYPE_F32
typedef float real;
#endif

#ifdef REAL_TYPE_F64
typedef double real;
#endif

typedef unsigned int uint;
typedef uint cell_id;

struct R3Nptrs {
    real *x;
    real *y;
    real *z;
    long int n;
};

struct R3Nvecs{
    thrust::device_vector<real> Xs; 
    thrust::device_vector<real> Ys; 
    thrust::device_vector<real> Zs;
};

struct hostR3NVec{
    thrust::host_vector<real> Xs; 
    thrust::host_vector<real> Ys; 
    thrust::host_vector<real> Zs;
};

struct real3{
    real x;
    real y;
    real z;
};

struct angles3{
    real aij, ajk, aik;
};


struct bounds_struct{
    real xMin, xMax;
    real yMin, yMax;
    real zMin, zMax; 
};


struct cell_bounds_struct{
    real *xMins;
    real *xMaxs;
    real *yMins;
    real *yMaxs;
    real *zMins;
    real *zMaxs;
};

typedef struct R3Nptrs R3Nptrs;
typedef struct R3Nvecs R3Nvecs;
typedef struct hostR3NVecs hostR3NVecs; 
typedef struct real3 real3;
typedef struct angles3 angles3;
typedef struct bounds_struct bounds_struct;
typedef struct uint3 uint3;
typedef struct cell_bounds_struct cell_bounds_struct; 





//State

// struct sim_state_struct{
//     R3Nptrs pos;
//     R3Nptrs vels; 
//     R3Nptrs forces;
//     R3Nptrs CoMs; 
//     uint no_of_cells;
//     uint t_step;
//     int *num_divs; 
// };

// typedef struct sim_state_struct state_struct;
#endif // TYPES_CUH

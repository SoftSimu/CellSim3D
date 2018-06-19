#ifndef RANDOM_VECTOR_H
#define RANDOM_VECTOR_H
#ifndef STD_LIB
#include <math.h>
#include <stdio.h>
#endif
#include "marsaglia.h"
#include "Types.cuh"

real3 GetRandomVector(){
    float ran2[2];
    float s = 0;
    do {
        ranmar(ran2, 2);
        ran2[0] = 2.0f*ran2[0] - 1.0f;
        ran2[1] = 2.0f*ran2[1] - 1.0f;
        s = ran2[0]*ran2[0] + ran2[1]*ran2[1];
    } while (s >= 1.0f);

    float x1 = ran2[0];
    float x2 = ran2[1];

    return (make_real3(2*x1*sqrt(1 - x1*x1 - x2*x2),
                       2*x2*sqrt(1 - x1*x1 - x2*x2),
                       1 - 2*(x1*x1 + x2*x2)));
}

real3 GetRandomVectorBasis (real3 _basis){
    // Credit for algorithm to Arthur Vromans, Sept 10, 2015, TUE CASA
    float ran[1];
    ranmar(ran, 1);
    float Nv = cos(2*3.14159*ran[0]);
    float Nw = sin(2*3.14159*ran[0]);
    float v[3], w[3];
    float basis[3] = {_basis.x, _basis.y, _basis.z};
    float n[3] = {0,0,0};

    if (basis[1] != 0){
        v[0] = 0;
        v[1] = basis[2];
        v[2] = -1*basis[1];

        w[0] = basis[1];
        w[1] = -1*basis[0];
        w[2] = 0;
    }else{ // this branch is very unlikely, placed for correctness
        v[0] = 0;
        v[1] = 1;
        v[2] = 0;

        w[0] = basis[2];
        w[1] = 0;
        w[2] = -1*basis[0];
    }

    // Orthogonalize
    float f = (w[0]*v[0] + w[1]*v[1] + w[2]*w[2])/
              (v[0]*v[0] + v[1]*v[1] + v[2]*v[2]);

    w[0] = w[0] - f*v[0];
    w[1] = w[1] - f*v[1];
    w[2] = w[2] - f*v[2];

    // normalize
    f = sqrt(v[0]*v[0] + v[1]*v[1] + v[2]*v[2]);

    v[0] = v[0]/f;
    v[1] = v[1]/f;
    v[2] = v[2]/f;

    f = sqrt(w[0]*w[0] + w[1]*w[1] + w[2]*w[2]);

    w[0] = w[0]/f;
    w[1] = w[1]/f;
    w[2] = w[2]/f;

    // Normal can be calculated:
    n[0] = Nv*v[0] + Nw*w[0];
    n[1] = Nv*v[1] + Nw*w[1];
    n[2] = Nv*v[2] + Nw*w[2];
    return (make_real3(n[0], n[1], n[2]));
}
#endif // RANDOM_VECTOR_H End

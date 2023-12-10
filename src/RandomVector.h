#ifndef RANDOM_VECTOR_H
#define RANDOM_VECTOR_H
#ifndef STD_LIB
#include <math.h>
#include <stdio.h>
#endif
#include "marsaglia.h"

void GetRandomVector (float* n){
    
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

    n[0] = 2*x1*sqrt(1 - x1*x1 - x2*x2);
    n[1] = 2*x2*sqrt(1 - x1*x1 - x2*x2);
    n[2] = 1 - 2*(x1*x1 + x2*x2);

}

void GetRandomVectorBasis (float* n, float* v, float* w){
    // Credit for algorithm to Arthur Vromans, Sept 10, 2015, TUE CASA
    
    float ran[1];
    ranmar(ran, 1);
    float Nv = cos(2*3.14159*ran[0]);
    float Nw = sin(2*3.14159*ran[0]);

    // Normal can be calculated:
    n[0] = Nv*v[0] + Nw*w[0];
    n[1] = Nv*v[1] + Nw*w[1];
    n[2] = Nv*v[2] + Nw*w[2];

}
#endif // RANDOM_VECTOR_H End

#!/bin/bash

sed -i 's/"trap;"/"s_trap 0x02;"/g' src/volume.cu
sed -i 's/"trap;"/"s_trap 0x02;"/g' src/propagate.cu
find . -type f \( -iname \*.cu -o -iname \*.cuh -o -iname \*.cpp -o -iname \*.hpp -o -iname \*.h -o -iname *.cxx \) -exec hipify-perl -inplace -experimental -print-stats {} \;

sed -i '1s/^/#include \"hip\/hip_runtime.h\" /' src/postscriptinit.cu 


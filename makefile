compiler = /usr/local/cuda/bin/nvcc
flags = -arch=sm_30 -Xptxas="-v" -c -O3 -g -I inc

objects = GPUbounce.o postscriptinit.o propagatebound.o centermass.o volume.o jsoncpp.o

CellDiv: $(objects)
	$(compiler) -O3 -arch=sm_30 -o "CellDiv" $(objects) -lm

GPUbounce.o: GPUbounce.cu postscript.h
	$(compiler) $(flags) GPUbounce.cu

postscriptinit.o: postscriptinit.cu postscript.h
	$(compiler) $(flags) postscriptinit.cu

propagatebound.o: propagatebound.cu postscript.h
	$(compiler) $(flags) propagatebound.cu

centermass.o: centermass.cu postscript.h
	$(compiler) $(flags) centermass.cu

volume.o: volume.cu postscript.h
	$(compiler) $(flags) volume.cu

jsoncpp.o: src/utils/jsoncpp.cpp inc/json/json.h
	$(compiler) $(flags) src/utils/jsoncpp.cpp

.PHONY: clean
clean:
	rm -f CellDiv $(objects)

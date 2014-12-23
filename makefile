compiler = $(shell which nvcc)
debug = -g -G
arch = -arch=sm_30
oflags = $(arch) -Xptxas="-v" -c -O3 -I inc

objects = GPUbounce.o postscriptinit.o propagatebound.o centermass.o volume.o jsoncpp.o

eflags = -O3 $(arch) -o "CellDiv" $(objects) -lm

debug: oflags += $(debug)
debug: eflags += $(debug)
debug: CellDiv

CellDiv: $(objects)
	$(compiler) $(eflags)

GPUbounce.o: GPUbounce.cu postscript.h
	$(compiler) $(oflags) GPUbounce.cu

postscriptinit.o: postscriptinit.cu postscript.h
	$(compiler) $(oflags) postscriptinit.cu

propagatebound.o: propagatebound.cu postscript.h
	$(compiler) $(oflags) propagatebound.cu

centermass.o: centermass.cu postscript.h
	$(compiler) $(oflags) centermass.cu

volume.o: volume.cu postscript.h
	$(compiler) $(oflags) volume.cu

jsoncpp.o: src/utils/jsoncpp.cpp inc/json/json.h
	$(compiler) $(oflags) src/utils/jsoncpp.cpp

.PHONY: clean
clean:
	rm -f CellDiv $(objects)

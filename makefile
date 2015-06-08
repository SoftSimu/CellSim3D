compiler = $(shell which nvcc)
debug = -g -G
arch = -arch=sm_30
oflags = $(arch) -Xptxas="-v" -c -O3 -I inc -lnetcdf

objects = GPUbounce.o postscriptinit.o propagatebound.o centermass.o volume.o jsoncpp.o PressureKernels.o BinaryOutput.o

eflags = -O3 $(arch) -o "CellDiv" $(objects) -lm -lnetcdf

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

PressureKernels.o: PressureKernels.cu postscript.h
	$(compiler) $(oflags) PressureKernels.cu

BinaryOutput.o: BinaryOutput.cu BinaryOutput.h
	$(compiler) $(oflags) BinaryOutput.cu

.PHONY: clean
clean:
	rm -f CellDiv $(objects)

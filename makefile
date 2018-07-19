compiler = $(shell which nvcc)
debug = -g -G
arch = -arch=sm_35
oflags = $(arch) -I inc -dc -D_FORCE_INLINES -I /usr/include/hdf5/serial/ -std=c++11 -lineinfo
objDir = bin/
sources = $(wildcard src/*.cu)
#objects = $(patsubst src%, $(objDir)%, $(patsubst %.cu, %.o, $(sources)))
objects = GPUbounce.o centermass.o postscriptinit.o PressureKernels.o\
	propagatebound.o propagate.o volume.o SimParams.o TrajWriter.o
linkObjects = $(patsubst %, $(objDir)%, $(objects))

eflags = $(arch) -o $(objDir)/"CellDiv" $(linkObjects) bin/jsoncpp.o -lm -lcurand -L/usr/lib/x86_64-linux-gnu/hdf5/serial -lhdf5 -std=c++11 -lineinfo
opt = -O3

ROOT = $(shell pwd)
rc = export CELLSIM3D_ROOT=$(ROOT)

debug: opt= -O0
debug: oflags += $(debug)
debug: eflags += $(debug)
debug: CellDiv

oflags += $(opt)
eflags += $(opt)

# $(objects): bin/%.o : src/%.cu
# 	@mkdir -p $(@D)
# 	$(compiler) $(oflags) -c $< -o $@

$(objDir)centermass.o: src/centermass.cu
	$(compiler) $(oflags) -c src/centermass.cu -o $(objDir)centermass.o

# NeighbourSearch.o: src/NeighbourSearch.cu
# 	$(compiler) $(oflags) -c src/NeighbourSearch.o

$(objDir)postscriptinit.o: src/postscriptinit.cu
	$(compiler) $(oflags) -c src/postscriptinit.cu -o $(objDir)postscriptinit.o

$(objDir)PressureKernels.o: src/PressureKernels.cu
	$(compiler) $(oflags) -c src/PressureKernels.cu -o $(objDir)PressureKernels.o

$(objDir)propagatebound.o: src/propagatebound.cu
	$(compiler) $(oflags) -c src/propagatebound.cu -o $(objDir)propagatebound.o

$(objDir)propagate.o: src/propagate.cu
	$(compiler) $(oflags) -c src/propagate.cu -o $(objDir)propagate.o

$(objDir)volume.o : src/volume.cu
	$(compiler) $(oflags) -c src/volume.cu -o $(objDir)volume.o

$(objDir)GPUbounce.o : src/GPUbounce.cu
	$(compiler) $(oflags) -c src/GPUbounce.cu -o $(objDir)GPUbounce.o

$(objDir)SimParams.o: src/SimParams.cu
	$(compiler) $(oflags) -c src/SimParams.cu -o $(objDir)SimParams.o

$(objDir)TrajWriter.o: src/TrajWriter.cu
	$(compiler) $(oflags) -c src/TrajWriter.cu -o $(objDir)TrajWriter.o

env:
	echo "$(rc)" > bin/CellSim3D.rc
	echo "export PATH=\$$PATH:$(ROOT)/bin" >> bin/CellSim3D.rc

CellDiv: $(linkObjects) $(objDir)jsoncpp.o env
	$(compiler) $(eflags)

install: CellDiv
	mkdir -p $(ROOT)
	cp -r --parents `git ls-files` $(ROOT)
	cp -r --parents bin $(ROOT)

# Third party libraries
$(objDir)jsoncpp.o: src/utils/jsoncpp.cpp inc/json/json.h
	$(compiler) $(oflags) -c src/utils/jsoncpp.cpp -o $(objDir)/jsoncpp.o

.PHONY: clean
clean:
	rm -f $(objDir)/*

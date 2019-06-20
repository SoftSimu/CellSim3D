compiler = $(shell which nvcc)
debug = -g -G -lineinfo
arch = -arch=sm_35
oflags = $(arch) -Xptxas="-v" -I inc -dc -D_FORCE_INLINES
objDir = bin/
sources = $(wildcard src/*.cu)
#objects = $(patsubst src%, $(objDir)%, $(patsubst %.cu, %.o, $(sources)))
objects = GPUbounce.o centermass.o postscriptinit.o PressureKernels.o\
	propagatebound.o propagate.o volume.o BondKernels.o
linkObjects = $(patsubst %, $(objDir)%, $(objects))

eflags = $(arch) -o $(objDir)/"CellDiv" $(linkObjects) bin/jsoncpp.o -lm -lcurand
opt = -O3

CELLSIM3D_ROOT = $(PWD)

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

$(objDir)BondKernels.o : src/BondKernels.cu
	$(compiler) $(oflags) -c src/BondKernels.cu -o $(objDir)BondKernels.o

$(objDir)GPUbounce.o : src/GPUbounce.cu
	$(compiler) $(oflags) -c src/GPUbounce.cu -o $(objDir)GPUbounce.o

CellDiv: $(linkObjects)
	$(compiler) $(eflags)
	echo "export PYTHONPATH=${CELLSIM3D_ROOT}/scripts/:\$$PYTHONPATH" > $(objDir)CellSim3D.rc
	echo "export PATH=${CELLSIM3D_ROOT}/bin:\$$PATH" >> $(objDir)CellSim3D.rc
	echo "${CELLSIM3D_ROOT}"


# Third party libraries
$(objDir)jsoncpp.o: src/utils/jsoncpp.cpp inc/json/json.h
	$(compiler) $(oflags) -c src/utils/jsoncpp.cpp -o $(objDir)/jsoncpp.o

.PHONY: clean
clean:
	rm -f $(objDir)/CellDiv $(linkObjects)

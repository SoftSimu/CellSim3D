compiler = $(shell which nvcc)
debug = -g -G -lineinfo
arch = -arch=sm_80
oflags = $(arch) -Xptxas="-v" -I inc -dc -lmpi
objDir = bin/
sources = $(wildcard src/*.cu)
#objects = $(patsubst src%, $(objDir)%, $(patsubst %.cu, %.o, $(sources)))

objects1 = GPUbounce.o centermass.o postscriptinit.o PressureKernels.o\
	propagatebound.o propagate.o volume.o BondKernels.o jsoncpp.o

objects2 = GPUbounce-CudaAwareMPI.o centermass.o postscriptinit.o PressureKernels.o\
	propagatebound.o propagate.o volume.o BondKernels.o jsoncpp.o


linkObjects1 = $(patsubst %, $(objDir)%, $(objects1))
linkObjects2 = $(patsubst %, $(objDir)%, $(objects2))

eflags1 = $(arch) -o $(objDir)/"CellDiv" $(linkObjects1) -lm -lcurand -lmpi
eflags2 = $(arch) -o $(objDir)/"CellDiv_CudaAwareMPI" $(linkObjects2) -lm -lcurand -lmpi
opt = -O3

debug: opt= -O0
debug: oflags += $(debug)
debug: eflags1 += $(debug)
debug: eflags2 += $(debug)
debug: CellDiv
debug: CellDiv_CudaAwareMPI

oflags += $(opt)
eflags1 += $(opt)
eflags2 += $(opt)

# $(objects): bin/%.o : src/%.cu
# 	@mkdir -p $(@D)
# 	$(compiler) $(oflags) -c $< -o $@


$(objDir)centermass.o: src/centermass.cu src/postscript.h
	$(compiler) $(oflags) -c src/centermass.cu -o $(objDir)centermass.o

$(objDir)postscriptinit.o: src/postscriptinit.cu
	$(compiler) $(oflags) -c src/postscriptinit.cu -o $(objDir)postscriptinit.o

$(objDir)PressureKernels.o: src/PressureKernels.cu src/postscript.h
	$(compiler) $(oflags) -c src/PressureKernels.cu -o $(objDir)PressureKernels.o

$(objDir)propagatebound.o: src/propagatebound.cu src/postscript.h
	$(compiler) $(oflags) -c src/propagatebound.cu -o $(objDir)propagatebound.o

$(objDir)propagate.o: src/propagate.cu
	$(compiler) $(oflags) -c src/propagate.cu -o $(objDir)propagate.o

$(objDir)volume.o : src/volume.cu
	$(compiler) $(oflags) -c src/volume.cu -o $(objDir)volume.o

$(objDir)BondKernels.o : src/BondKernels.cu
	$(compiler) $(oflags) -c src/BondKernels.cu -o $(objDir)BondKernels.o

$(objDir)GPUbounce.o : src/GPUbounce.cu src/postscript.h
	$(compiler) $(oflags) -c src/GPUbounce.cu -o $(objDir)GPUbounce.o

$(objDir)GPUbounce-CudaAwareMPI.o : src/GPUbounce-CudaAwareMPI.cu src/postscript.h
	$(compiler) $(oflags) -c src/GPUbounce-CudaAwareMPI.cu -o $(objDir)GPUbounce-CudaAwareMPI.o


CellDiv: $(linkObjects1)
	$(compiler) $(eflags1)

CellDiv_CudaAwareMPI: $(linkObjects2)
	$(compiler) $(eflags2)

# Third party libraries
$(objDir)jsoncpp.o: src/utils/jsoncpp.cpp inc/json/json.h
	$(compiler) $(oflags) -c src/utils/jsoncpp.cpp -o $(objDir)/jsoncpp.o

.PHONY: clean
clean:
	rm -f $(objDir)/CellDiv $(linkObjects1)

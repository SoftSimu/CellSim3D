compiler = nvcc
flags = -arch=sm_30 -Xptxas="-v" -c -O3 

objects = GPUbounce.o marsaglia.o postscriptinit.o propagatebound.o centermass.o volume.o

CellDiv: $(objects)
	$(compiler) -O3 -arch=sm_30 -o "CellDiv" $(objects) -lm

GPUbounce.o: GPUbounce.cu postscript.h
	$(compiler) $(flags) GPUbounce.cu

marsaglia.o: marsaglia.cu postscript.h
	$(compiler) $(flags) marsaglia.cu

postscriptinit.o: postscriptinit.cu postscript.h
	$(compiler) $(flags) postscriptinit.cu

propagatebound.o: propagatebound.cu postscript.h
	$(compiler) $(flags) propagatebound.cu

centermass.o: centermass.cu postscript.h
	$(compiler) $(flags) centermass.cu

volume.o: volume.cu postscript.h
	$(compiler) $(flags) volume.cu

.PHONY: clean
clean:
	rm -f CellDiv $(objects)

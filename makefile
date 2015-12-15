compiler = $(shell which nvcc)
debug = -g -G
arch = -arch=sm_35
oflags = $(arch) -Xptxas="-v" -O3 -I inc
objDir = bin
sources = $(wildcard src/*.cu)
objects = $(patsubst src%, $(objDir)%, $(patsubst %.cu, %.o, $(sources)))

eflags = -O3 $(arch) -o $(objDir)/"CellDiv" $(objects) bin/jsoncpp.o -lm

debug: oflags += $(debug)
debug: eflags += $(debug)
debug: CellDiv

$(objects): bin/%.o : src/%.cu
	@mkdir -p $(@D)
	$(compiler) $(oflags) -c $< -o $@

CellDiv: $(objects) jsoncpp.o
	$(compiler) $(eflags)

# Third party libraries
jsoncpp.o: src/utils/jsoncpp.cpp inc/json/json.h
	$(compiler) $(oflags) -c src/utils/jsoncpp.cpp -o $(objDir)/jsoncpp.o

.PHONY: clean
clean:
	rm -f $(objDir)/CellDiv $(objects)

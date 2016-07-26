compiler = $(shell which nvcc)
debug = -g -G -lineinfo
arch = -arch=sm_52
oflags = $(arch) -Xptxas="-v" -I inc -dc
objDir = bin
sources = $(wildcard src/*.cu)
objects = $(patsubst src%, $(objDir)%, $(patsubst %.cu, %.o, $(sources)))

eflags = $(arch) -o $(objDir)/"CellDiv" $(objects) bin/jsoncpp.o -lm -lcurand
opt = -O3

debug: opt= -O0
debug: oflags += $(debug)
debug: eflags += $(debug)
debug: CellDiv

oflags += $(opt)
eflags += $(opt)

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

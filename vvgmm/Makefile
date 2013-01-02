UNAME=$(shell uname)

ifeq ($(UNAME), Darwin)
LD_CBLAS=-framework accelerate
endif
ifeq ($(UNAME), Linux)
LD_CBLAS=-lblas
endif

CXX_FLAGS=-std=c++0x -Wall -pedantic -I.
LD_FLAGS=-lglog -lgflags $(LD_CBLAS)
BINARIES=vvgmm-train

all: $(BINARIES)

gmm.o: gmm.cc gmm.h
	$(CXX) -c $< $(CXX_FLAGS)

vvgmm-train.o: vvgmm-train.cc
	$(CXX) -c $< $(CXX_FLAGS)

vvgmm-train: vvgmm-train.o gmm.o
	$(CXX) -o $@ $^ $(LD_FLAGS)

clean:
	rm -f *.o *~

distclean: clean
	rm -f $(BINARIES)
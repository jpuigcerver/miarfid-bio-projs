# Libraries depending on the operating system
UNAME=$(shell uname)
ifeq ($(UNAME), Darwin)
LD_OS=-framework accelerate
endif
ifeq ($(UNAME), Linux)
LD_OS=-lblas
endif

CXX_FLAGS=-std=c++0x -Wall -pedantic -I./ `Magick++-config --cxxflags` \
-DNDEBUG -O3
LD_FLAGS=-lgflags -lglog -lprotobuf `Magick++-config --ldflags --libs` \
$(LD_OS)
BINARIES=sk-train sk-test

all: prot $(BINARIES)

prot:
	make -C protos

dataset.o: dataset.cc dataset.h
	$(CXX) -c $< $(CXX_FLAGS)

utils.o: utils.cc utils.h
	$(CXX) -c $< $(CXX_FLAGS)

sk-model.o: sk-model.cc sk-model.h
	$(CXX) -c $< $(CXX_FLAGS)

k-clustering.o: k-clustering.cc k-clustering.h
	$(CXX) -c $< $(CXX_FLAGS)

sk-train.o: sk-train.cc
	$(CXX) -c $< $(CXX_FLAGS)

sk-train: sk-train.o dataset.o utils.o sk-model.o k-clustering.o pca.o \
	protos/sk-model.pb.o protos/k-clustering.pb.o
	$(CXX) -o $@ $^  $(LD_FLAGS)

sk-test.o: sk-test.cc
	$(CXX) -c $< $(CXX_FLAGS)

sk-test: sk-test.o dataset.o utils.o sk-model.o k-clustering.o pca.o \
	protos/sk-model.pb.o protos/k-clustering.pb.o
	$(CXX) -o $@ $^  $(LD_FLAGS)

test-clustering.o: test-clustering.cc
	$(CXX) -c $< $(CXX_FLAGS)

test-clustering: test-clustering.o utils.o k-clustering.o protos/k-clustering.pb.o
	$(CXX) -o $@ $^  $(LD_FLAGS)

pca.o: pca.cc
	$(CXX) -c $< $(CXX_FLAGS)

clean:
	make -C protos clean
	rm -f *.o *~

distclean: clean
	rm -f $(BINARIES)
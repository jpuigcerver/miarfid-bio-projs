CXX_FLAGS=-std=c++0x -Wall -pedantic `Magick++-config  --cppflags` -I./ -g
LD_FLAGS=-lgflags -lglog -lblas -lprotobuf `Magick++-config --ldflags --libs` -g
BINARIES=sk-train

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

sk-train: sk-train.o dataset.o utils.o sk-model.o k-clustering.o \
	protos/sk-model.pb.o protos/k-clustering.pb.o
	$(CXX) -o $@ $^  $(LD_FLAGS)

clean:
	make -C protos clean
	rm -f *.o *~

distclean: clean
	rm -f $(BINARIES)
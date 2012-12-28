BINARIES=sk
CXX_FLAGS=-std=c++11 -Wall -pedantic `Magick++-config  --cppflags` -I./
LD_FLAGS=-lgflags -lglog `Magick++-config --ldflags --libs`

all: $(BINARIES)

dataset.o: dataset.cc dataset.h
	$(CXX) -c $< $(CXX_FLAGS)

sk.o: sk.cc
	$(CXX) -c $< $(CXX_FLAGS)

sk: sk.o dataset.o
	$(CXX) -o $@ $^ $(LD_FLAGS)

sk_model.o: sk_model.cc sk_model.h
	$(CXX) -c $< $(CXX_FLAGS)

k-clustering.o: k-clustering.cc k-clustering.h
	$(CXX) -c $< $(CXX_FLAGS)

clean:
	rm -f *.o *~

distclean: clean
	rm -f $(BINARIES)
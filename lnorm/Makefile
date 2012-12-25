CXX_FLAGS=-std=c++11 -Wall -pedantic `Magick++-config  --cxxflags` -I./ -pg -g
LD_FLAGS=-lgflags -lglog `Magick++-config --ldflags --libs`
BINARIES=lnorm

all: $(BINARIES)

lnorm.o: lnorm.cc
	$(CXX) -c $< $(CXX_FLAGS)

lnorm: lnorm.o
	$(CXX) -o $@ $^ $(LD_FLAGS)

clean:
	rm -f *.o *~

distclean: clean
	rm -f $(BINARIES)

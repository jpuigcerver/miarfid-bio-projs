CXX_FLAGS=-std=c++0x -Wall -pedantic `Magick++-config  --cppflags` -I./ \
-O3 -DNDEBUG
LD_FLAGS=-lgflags -lglog `Magick++-config --ldflags --libs`
BINARIES=lhequ

all: $(BINARIES)

lhequ.o: lhequ.cc
	$(CXX) -c $< $(CXX_FLAGS)

lhequ: lhequ.o
	$(CXX) -o $@ $^ $(LD_FLAGS)

clean:
	rm -f *.o *~

distclean: clean
	rm -f $(BINARIES)

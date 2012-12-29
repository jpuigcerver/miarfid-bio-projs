PROTO_PATH=.
CPP_OUT=.
CXX_FLAGS=-Wall -pedantic -std=c++0x -O3

all: k-clustering sk-model

k-clustering: k-clustering.proto
	protoc --proto_path=$(PROTO_PATH) --cpp_out=$(CPP_OUT) $<
	$(CXX) -c $(CPP_OUT)/$@.pb.cc $(CXX_FLAGS)

sk-model: sk-model.proto
	protoc --proto_path=$(PROTO_PATH) --cpp_out=$(CPP_OUT) $<
	$(CXX) -c $(CPP_OUT)/$@.pb.cc $(CXX_FLAGS)

clean:
	rm -f *.cc *.h *.o
BUILD_DEBUG = OFF
CMAKE_FLAGS = -DBUILD_DEBUG=$(BUILD_DEBUG)

all: nn_control_server

.PHONY: debug
debug:
	$(eval BUILD_DEBUG=ON)

.PHONY: nn_control_server
nn_control_server:
	export CC=clang && export CXX=clang++ && mkdir -p build && cd build && cmake -DBUILD_MAIN=ON $(CMAKE_FLAGS) -G Ninja .. && ulimit -Sv 6500000 && ninja -j1

.PHONY: nn_control_server_verbose
nn_control_server_verbose:
	export CC=clang && export CXX=clang++ && mkdir -p build && cd build && cmake -DBUILD_MAIN=ON $(CMAKE_FLAGS) -G Ninja .. && ulimit -Sv 6500000 && ninja -j1 VERBOSE=1

.PHONY: no_main
no_main:
	export CC=clang && export CXX=clang++ && mkdir -p build && cd build && cmake -DBUILD_MAIN=OFF $(CMAKE_FLAGS) -G Ninja .. && ulimit -Sv 6500000 && ninja -j1

.PHONY: clean 
clean:
	rm -rf build

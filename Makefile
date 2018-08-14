all: nn_control_server

nn_control_server:
	export CC=clang && export CXX=clang++ && mkdir -p build && cd build && cmake -DBUILD_MAIN=ON -G Ninja .. && ulimit -Sv 6500000 && ninja -j1

nn_control_server_verbose:
	export CC=clang && export CXX=clang++ && mkdir -p build && cd build && cmake -DBUILD_MAIN=ON -G Ninja .. && ulimit -Sv 6500000 && ninja -j1 VERBOSE=1

no_main:
	export CC=clang && export CXX=clang++ && mkdir -p build && cd build && cmake -DBUILD_MAIN=OFF -G Ninja .. && ulimit -Sv 6500000 && ninja -j1

clean:
	rm -rf build

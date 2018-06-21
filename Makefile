all: nn_control_server

nn_control_server:
	mkdir -p build && cd build && cmake ../ && make

clean:
	rm -rf build

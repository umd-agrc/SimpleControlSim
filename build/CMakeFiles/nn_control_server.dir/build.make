# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.4

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:


#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:


# Remove some rules from gmake that .SUFFIXES does not remove.
SUFFIXES =

.SUFFIXES: .hpux_make_needs_suffix_list


# Suppress display of executed commands.
$(VERBOSE).SILENT:


# A target that is always out of date.
cmake_force:

.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /opt/cmake-3.4.3-Linux-x86_64/bin/cmake

# The command to remove a file.
RM = /opt/cmake-3.4.3-Linux-x86_64/bin/cmake -E remove -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/eric/Programs/Snapdragon/nn_control

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/eric/Programs/Snapdragon/nn_control/build

# Include any dependencies generated for this target.
include CMakeFiles/nn_control_server.dir/depend.make

# Include the progress variables for this target.
include CMakeFiles/nn_control_server.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/nn_control_server.dir/flags.make

CMakeFiles/nn_control_server.dir/trainer.cpp.o: CMakeFiles/nn_control_server.dir/flags.make
CMakeFiles/nn_control_server.dir/trainer.cpp.o: ../trainer.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/eric/Programs/Snapdragon/nn_control/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/nn_control_server.dir/trainer.cpp.o"
	/usr/bin/c++   $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/nn_control_server.dir/trainer.cpp.o -c /home/eric/Programs/Snapdragon/nn_control/trainer.cpp

CMakeFiles/nn_control_server.dir/trainer.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/nn_control_server.dir/trainer.cpp.i"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/eric/Programs/Snapdragon/nn_control/trainer.cpp > CMakeFiles/nn_control_server.dir/trainer.cpp.i

CMakeFiles/nn_control_server.dir/trainer.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/nn_control_server.dir/trainer.cpp.s"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/eric/Programs/Snapdragon/nn_control/trainer.cpp -o CMakeFiles/nn_control_server.dir/trainer.cpp.s

CMakeFiles/nn_control_server.dir/trainer.cpp.o.requires:

.PHONY : CMakeFiles/nn_control_server.dir/trainer.cpp.o.requires

CMakeFiles/nn_control_server.dir/trainer.cpp.o.provides: CMakeFiles/nn_control_server.dir/trainer.cpp.o.requires
	$(MAKE) -f CMakeFiles/nn_control_server.dir/build.make CMakeFiles/nn_control_server.dir/trainer.cpp.o.provides.build
.PHONY : CMakeFiles/nn_control_server.dir/trainer.cpp.o.provides

CMakeFiles/nn_control_server.dir/trainer.cpp.o.provides.build: CMakeFiles/nn_control_server.dir/trainer.cpp.o


CMakeFiles/nn_control_server.dir/simulator.cpp.o: CMakeFiles/nn_control_server.dir/flags.make
CMakeFiles/nn_control_server.dir/simulator.cpp.o: ../simulator.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/eric/Programs/Snapdragon/nn_control/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Building CXX object CMakeFiles/nn_control_server.dir/simulator.cpp.o"
	/usr/bin/c++   $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/nn_control_server.dir/simulator.cpp.o -c /home/eric/Programs/Snapdragon/nn_control/simulator.cpp

CMakeFiles/nn_control_server.dir/simulator.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/nn_control_server.dir/simulator.cpp.i"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/eric/Programs/Snapdragon/nn_control/simulator.cpp > CMakeFiles/nn_control_server.dir/simulator.cpp.i

CMakeFiles/nn_control_server.dir/simulator.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/nn_control_server.dir/simulator.cpp.s"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/eric/Programs/Snapdragon/nn_control/simulator.cpp -o CMakeFiles/nn_control_server.dir/simulator.cpp.s

CMakeFiles/nn_control_server.dir/simulator.cpp.o.requires:

.PHONY : CMakeFiles/nn_control_server.dir/simulator.cpp.o.requires

CMakeFiles/nn_control_server.dir/simulator.cpp.o.provides: CMakeFiles/nn_control_server.dir/simulator.cpp.o.requires
	$(MAKE) -f CMakeFiles/nn_control_server.dir/build.make CMakeFiles/nn_control_server.dir/simulator.cpp.o.provides.build
.PHONY : CMakeFiles/nn_control_server.dir/simulator.cpp.o.provides

CMakeFiles/nn_control_server.dir/simulator.cpp.o.provides.build: CMakeFiles/nn_control_server.dir/simulator.cpp.o


CMakeFiles/nn_control_server.dir/data.cpp.o: CMakeFiles/nn_control_server.dir/flags.make
CMakeFiles/nn_control_server.dir/data.cpp.o: ../data.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/eric/Programs/Snapdragon/nn_control/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_3) "Building CXX object CMakeFiles/nn_control_server.dir/data.cpp.o"
	/usr/bin/c++   $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/nn_control_server.dir/data.cpp.o -c /home/eric/Programs/Snapdragon/nn_control/data.cpp

CMakeFiles/nn_control_server.dir/data.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/nn_control_server.dir/data.cpp.i"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/eric/Programs/Snapdragon/nn_control/data.cpp > CMakeFiles/nn_control_server.dir/data.cpp.i

CMakeFiles/nn_control_server.dir/data.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/nn_control_server.dir/data.cpp.s"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/eric/Programs/Snapdragon/nn_control/data.cpp -o CMakeFiles/nn_control_server.dir/data.cpp.s

CMakeFiles/nn_control_server.dir/data.cpp.o.requires:

.PHONY : CMakeFiles/nn_control_server.dir/data.cpp.o.requires

CMakeFiles/nn_control_server.dir/data.cpp.o.provides: CMakeFiles/nn_control_server.dir/data.cpp.o.requires
	$(MAKE) -f CMakeFiles/nn_control_server.dir/build.make CMakeFiles/nn_control_server.dir/data.cpp.o.provides.build
.PHONY : CMakeFiles/nn_control_server.dir/data.cpp.o.provides

CMakeFiles/nn_control_server.dir/data.cpp.o.provides.build: CMakeFiles/nn_control_server.dir/data.cpp.o


CMakeFiles/nn_control_server.dir/dynamics.cpp.o: CMakeFiles/nn_control_server.dir/flags.make
CMakeFiles/nn_control_server.dir/dynamics.cpp.o: ../dynamics.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/eric/Programs/Snapdragon/nn_control/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_4) "Building CXX object CMakeFiles/nn_control_server.dir/dynamics.cpp.o"
	/usr/bin/c++   $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/nn_control_server.dir/dynamics.cpp.o -c /home/eric/Programs/Snapdragon/nn_control/dynamics.cpp

CMakeFiles/nn_control_server.dir/dynamics.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/nn_control_server.dir/dynamics.cpp.i"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/eric/Programs/Snapdragon/nn_control/dynamics.cpp > CMakeFiles/nn_control_server.dir/dynamics.cpp.i

CMakeFiles/nn_control_server.dir/dynamics.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/nn_control_server.dir/dynamics.cpp.s"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/eric/Programs/Snapdragon/nn_control/dynamics.cpp -o CMakeFiles/nn_control_server.dir/dynamics.cpp.s

CMakeFiles/nn_control_server.dir/dynamics.cpp.o.requires:

.PHONY : CMakeFiles/nn_control_server.dir/dynamics.cpp.o.requires

CMakeFiles/nn_control_server.dir/dynamics.cpp.o.provides: CMakeFiles/nn_control_server.dir/dynamics.cpp.o.requires
	$(MAKE) -f CMakeFiles/nn_control_server.dir/build.make CMakeFiles/nn_control_server.dir/dynamics.cpp.o.provides.build
.PHONY : CMakeFiles/nn_control_server.dir/dynamics.cpp.o.provides

CMakeFiles/nn_control_server.dir/dynamics.cpp.o.provides.build: CMakeFiles/nn_control_server.dir/dynamics.cpp.o


CMakeFiles/nn_control_server.dir/diff.cpp.o: CMakeFiles/nn_control_server.dir/flags.make
CMakeFiles/nn_control_server.dir/diff.cpp.o: ../diff.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/eric/Programs/Snapdragon/nn_control/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_5) "Building CXX object CMakeFiles/nn_control_server.dir/diff.cpp.o"
	/usr/bin/c++   $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/nn_control_server.dir/diff.cpp.o -c /home/eric/Programs/Snapdragon/nn_control/diff.cpp

CMakeFiles/nn_control_server.dir/diff.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/nn_control_server.dir/diff.cpp.i"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/eric/Programs/Snapdragon/nn_control/diff.cpp > CMakeFiles/nn_control_server.dir/diff.cpp.i

CMakeFiles/nn_control_server.dir/diff.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/nn_control_server.dir/diff.cpp.s"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/eric/Programs/Snapdragon/nn_control/diff.cpp -o CMakeFiles/nn_control_server.dir/diff.cpp.s

CMakeFiles/nn_control_server.dir/diff.cpp.o.requires:

.PHONY : CMakeFiles/nn_control_server.dir/diff.cpp.o.requires

CMakeFiles/nn_control_server.dir/diff.cpp.o.provides: CMakeFiles/nn_control_server.dir/diff.cpp.o.requires
	$(MAKE) -f CMakeFiles/nn_control_server.dir/build.make CMakeFiles/nn_control_server.dir/diff.cpp.o.provides.build
.PHONY : CMakeFiles/nn_control_server.dir/diff.cpp.o.provides

CMakeFiles/nn_control_server.dir/diff.cpp.o.provides.build: CMakeFiles/nn_control_server.dir/diff.cpp.o


CMakeFiles/nn_control_server.dir/genann.c.o: CMakeFiles/nn_control_server.dir/flags.make
CMakeFiles/nn_control_server.dir/genann.c.o: ../genann.c
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/eric/Programs/Snapdragon/nn_control/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_6) "Building C object CMakeFiles/nn_control_server.dir/genann.c.o"
	/usr/bin/cc  $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -o CMakeFiles/nn_control_server.dir/genann.c.o   -c /home/eric/Programs/Snapdragon/nn_control/genann.c

CMakeFiles/nn_control_server.dir/genann.c.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing C source to CMakeFiles/nn_control_server.dir/genann.c.i"
	/usr/bin/cc  $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -E /home/eric/Programs/Snapdragon/nn_control/genann.c > CMakeFiles/nn_control_server.dir/genann.c.i

CMakeFiles/nn_control_server.dir/genann.c.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling C source to assembly CMakeFiles/nn_control_server.dir/genann.c.s"
	/usr/bin/cc  $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -S /home/eric/Programs/Snapdragon/nn_control/genann.c -o CMakeFiles/nn_control_server.dir/genann.c.s

CMakeFiles/nn_control_server.dir/genann.c.o.requires:

.PHONY : CMakeFiles/nn_control_server.dir/genann.c.o.requires

CMakeFiles/nn_control_server.dir/genann.c.o.provides: CMakeFiles/nn_control_server.dir/genann.c.o.requires
	$(MAKE) -f CMakeFiles/nn_control_server.dir/build.make CMakeFiles/nn_control_server.dir/genann.c.o.provides.build
.PHONY : CMakeFiles/nn_control_server.dir/genann.c.o.provides

CMakeFiles/nn_control_server.dir/genann.c.o.provides.build: CMakeFiles/nn_control_server.dir/genann.c.o


CMakeFiles/nn_control_server.dir/matrix_vector_ops.cpp.o: CMakeFiles/nn_control_server.dir/flags.make
CMakeFiles/nn_control_server.dir/matrix_vector_ops.cpp.o: ../matrix_vector_ops.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/eric/Programs/Snapdragon/nn_control/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_7) "Building CXX object CMakeFiles/nn_control_server.dir/matrix_vector_ops.cpp.o"
	/usr/bin/c++   $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/nn_control_server.dir/matrix_vector_ops.cpp.o -c /home/eric/Programs/Snapdragon/nn_control/matrix_vector_ops.cpp

CMakeFiles/nn_control_server.dir/matrix_vector_ops.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/nn_control_server.dir/matrix_vector_ops.cpp.i"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/eric/Programs/Snapdragon/nn_control/matrix_vector_ops.cpp > CMakeFiles/nn_control_server.dir/matrix_vector_ops.cpp.i

CMakeFiles/nn_control_server.dir/matrix_vector_ops.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/nn_control_server.dir/matrix_vector_ops.cpp.s"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/eric/Programs/Snapdragon/nn_control/matrix_vector_ops.cpp -o CMakeFiles/nn_control_server.dir/matrix_vector_ops.cpp.s

CMakeFiles/nn_control_server.dir/matrix_vector_ops.cpp.o.requires:

.PHONY : CMakeFiles/nn_control_server.dir/matrix_vector_ops.cpp.o.requires

CMakeFiles/nn_control_server.dir/matrix_vector_ops.cpp.o.provides: CMakeFiles/nn_control_server.dir/matrix_vector_ops.cpp.o.requires
	$(MAKE) -f CMakeFiles/nn_control_server.dir/build.make CMakeFiles/nn_control_server.dir/matrix_vector_ops.cpp.o.provides.build
.PHONY : CMakeFiles/nn_control_server.dir/matrix_vector_ops.cpp.o.provides

CMakeFiles/nn_control_server.dir/matrix_vector_ops.cpp.o.provides.build: CMakeFiles/nn_control_server.dir/matrix_vector_ops.cpp.o


# Object files for target nn_control_server
nn_control_server_OBJECTS = \
"CMakeFiles/nn_control_server.dir/trainer.cpp.o" \
"CMakeFiles/nn_control_server.dir/simulator.cpp.o" \
"CMakeFiles/nn_control_server.dir/data.cpp.o" \
"CMakeFiles/nn_control_server.dir/dynamics.cpp.o" \
"CMakeFiles/nn_control_server.dir/diff.cpp.o" \
"CMakeFiles/nn_control_server.dir/genann.c.o" \
"CMakeFiles/nn_control_server.dir/matrix_vector_ops.cpp.o"

# External object files for target nn_control_server
nn_control_server_EXTERNAL_OBJECTS =

nn_control_server: CMakeFiles/nn_control_server.dir/trainer.cpp.o
nn_control_server: CMakeFiles/nn_control_server.dir/simulator.cpp.o
nn_control_server: CMakeFiles/nn_control_server.dir/data.cpp.o
nn_control_server: CMakeFiles/nn_control_server.dir/dynamics.cpp.o
nn_control_server: CMakeFiles/nn_control_server.dir/diff.cpp.o
nn_control_server: CMakeFiles/nn_control_server.dir/genann.c.o
nn_control_server: CMakeFiles/nn_control_server.dir/matrix_vector_ops.cpp.o
nn_control_server: CMakeFiles/nn_control_server.dir/build.make
nn_control_server: CMakeFiles/nn_control_server.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/eric/Programs/Snapdragon/nn_control/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_8) "Linking CXX executable nn_control_server"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/nn_control_server.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/nn_control_server.dir/build: nn_control_server

.PHONY : CMakeFiles/nn_control_server.dir/build

CMakeFiles/nn_control_server.dir/requires: CMakeFiles/nn_control_server.dir/trainer.cpp.o.requires
CMakeFiles/nn_control_server.dir/requires: CMakeFiles/nn_control_server.dir/simulator.cpp.o.requires
CMakeFiles/nn_control_server.dir/requires: CMakeFiles/nn_control_server.dir/data.cpp.o.requires
CMakeFiles/nn_control_server.dir/requires: CMakeFiles/nn_control_server.dir/dynamics.cpp.o.requires
CMakeFiles/nn_control_server.dir/requires: CMakeFiles/nn_control_server.dir/diff.cpp.o.requires
CMakeFiles/nn_control_server.dir/requires: CMakeFiles/nn_control_server.dir/genann.c.o.requires
CMakeFiles/nn_control_server.dir/requires: CMakeFiles/nn_control_server.dir/matrix_vector_ops.cpp.o.requires

.PHONY : CMakeFiles/nn_control_server.dir/requires

CMakeFiles/nn_control_server.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/nn_control_server.dir/cmake_clean.cmake
.PHONY : CMakeFiles/nn_control_server.dir/clean

CMakeFiles/nn_control_server.dir/depend:
	cd /home/eric/Programs/Snapdragon/nn_control/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/eric/Programs/Snapdragon/nn_control /home/eric/Programs/Snapdragon/nn_control /home/eric/Programs/Snapdragon/nn_control/build /home/eric/Programs/Snapdragon/nn_control/build /home/eric/Programs/Snapdragon/nn_control/build/CMakeFiles/nn_control_server.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/nn_control_server.dir/depend


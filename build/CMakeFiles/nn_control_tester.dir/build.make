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
include CMakeFiles/nn_control_tester.dir/depend.make

# Include the progress variables for this target.
include CMakeFiles/nn_control_tester.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/nn_control_tester.dir/flags.make

CMakeFiles/nn_control_tester.dir/tester.cpp.o: CMakeFiles/nn_control_tester.dir/flags.make
CMakeFiles/nn_control_tester.dir/tester.cpp.o: ../tester.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/eric/Programs/Snapdragon/nn_control/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/nn_control_tester.dir/tester.cpp.o"
	/usr/bin/c++   $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/nn_control_tester.dir/tester.cpp.o -c /home/eric/Programs/Snapdragon/nn_control/tester.cpp

CMakeFiles/nn_control_tester.dir/tester.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/nn_control_tester.dir/tester.cpp.i"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/eric/Programs/Snapdragon/nn_control/tester.cpp > CMakeFiles/nn_control_tester.dir/tester.cpp.i

CMakeFiles/nn_control_tester.dir/tester.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/nn_control_tester.dir/tester.cpp.s"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/eric/Programs/Snapdragon/nn_control/tester.cpp -o CMakeFiles/nn_control_tester.dir/tester.cpp.s

CMakeFiles/nn_control_tester.dir/tester.cpp.o.requires:

.PHONY : CMakeFiles/nn_control_tester.dir/tester.cpp.o.requires

CMakeFiles/nn_control_tester.dir/tester.cpp.o.provides: CMakeFiles/nn_control_tester.dir/tester.cpp.o.requires
	$(MAKE) -f CMakeFiles/nn_control_tester.dir/build.make CMakeFiles/nn_control_tester.dir/tester.cpp.o.provides.build
.PHONY : CMakeFiles/nn_control_tester.dir/tester.cpp.o.provides

CMakeFiles/nn_control_tester.dir/tester.cpp.o.provides.build: CMakeFiles/nn_control_tester.dir/tester.cpp.o


CMakeFiles/nn_control_tester.dir/data.cpp.o: CMakeFiles/nn_control_tester.dir/flags.make
CMakeFiles/nn_control_tester.dir/data.cpp.o: ../data.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/eric/Programs/Snapdragon/nn_control/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Building CXX object CMakeFiles/nn_control_tester.dir/data.cpp.o"
	/usr/bin/c++   $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/nn_control_tester.dir/data.cpp.o -c /home/eric/Programs/Snapdragon/nn_control/data.cpp

CMakeFiles/nn_control_tester.dir/data.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/nn_control_tester.dir/data.cpp.i"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/eric/Programs/Snapdragon/nn_control/data.cpp > CMakeFiles/nn_control_tester.dir/data.cpp.i

CMakeFiles/nn_control_tester.dir/data.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/nn_control_tester.dir/data.cpp.s"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/eric/Programs/Snapdragon/nn_control/data.cpp -o CMakeFiles/nn_control_tester.dir/data.cpp.s

CMakeFiles/nn_control_tester.dir/data.cpp.o.requires:

.PHONY : CMakeFiles/nn_control_tester.dir/data.cpp.o.requires

CMakeFiles/nn_control_tester.dir/data.cpp.o.provides: CMakeFiles/nn_control_tester.dir/data.cpp.o.requires
	$(MAKE) -f CMakeFiles/nn_control_tester.dir/build.make CMakeFiles/nn_control_tester.dir/data.cpp.o.provides.build
.PHONY : CMakeFiles/nn_control_tester.dir/data.cpp.o.provides

CMakeFiles/nn_control_tester.dir/data.cpp.o.provides.build: CMakeFiles/nn_control_tester.dir/data.cpp.o


CMakeFiles/nn_control_tester.dir/testDynamics.cpp.o: CMakeFiles/nn_control_tester.dir/flags.make
CMakeFiles/nn_control_tester.dir/testDynamics.cpp.o: ../testDynamics.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/eric/Programs/Snapdragon/nn_control/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_3) "Building CXX object CMakeFiles/nn_control_tester.dir/testDynamics.cpp.o"
	/usr/bin/c++   $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/nn_control_tester.dir/testDynamics.cpp.o -c /home/eric/Programs/Snapdragon/nn_control/testDynamics.cpp

CMakeFiles/nn_control_tester.dir/testDynamics.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/nn_control_tester.dir/testDynamics.cpp.i"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/eric/Programs/Snapdragon/nn_control/testDynamics.cpp > CMakeFiles/nn_control_tester.dir/testDynamics.cpp.i

CMakeFiles/nn_control_tester.dir/testDynamics.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/nn_control_tester.dir/testDynamics.cpp.s"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/eric/Programs/Snapdragon/nn_control/testDynamics.cpp -o CMakeFiles/nn_control_tester.dir/testDynamics.cpp.s

CMakeFiles/nn_control_tester.dir/testDynamics.cpp.o.requires:

.PHONY : CMakeFiles/nn_control_tester.dir/testDynamics.cpp.o.requires

CMakeFiles/nn_control_tester.dir/testDynamics.cpp.o.provides: CMakeFiles/nn_control_tester.dir/testDynamics.cpp.o.requires
	$(MAKE) -f CMakeFiles/nn_control_tester.dir/build.make CMakeFiles/nn_control_tester.dir/testDynamics.cpp.o.provides.build
.PHONY : CMakeFiles/nn_control_tester.dir/testDynamics.cpp.o.provides

CMakeFiles/nn_control_tester.dir/testDynamics.cpp.o.provides.build: CMakeFiles/nn_control_tester.dir/testDynamics.cpp.o


CMakeFiles/nn_control_tester.dir/diff.cpp.o: CMakeFiles/nn_control_tester.dir/flags.make
CMakeFiles/nn_control_tester.dir/diff.cpp.o: ../diff.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/eric/Programs/Snapdragon/nn_control/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_4) "Building CXX object CMakeFiles/nn_control_tester.dir/diff.cpp.o"
	/usr/bin/c++   $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/nn_control_tester.dir/diff.cpp.o -c /home/eric/Programs/Snapdragon/nn_control/diff.cpp

CMakeFiles/nn_control_tester.dir/diff.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/nn_control_tester.dir/diff.cpp.i"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/eric/Programs/Snapdragon/nn_control/diff.cpp > CMakeFiles/nn_control_tester.dir/diff.cpp.i

CMakeFiles/nn_control_tester.dir/diff.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/nn_control_tester.dir/diff.cpp.s"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/eric/Programs/Snapdragon/nn_control/diff.cpp -o CMakeFiles/nn_control_tester.dir/diff.cpp.s

CMakeFiles/nn_control_tester.dir/diff.cpp.o.requires:

.PHONY : CMakeFiles/nn_control_tester.dir/diff.cpp.o.requires

CMakeFiles/nn_control_tester.dir/diff.cpp.o.provides: CMakeFiles/nn_control_tester.dir/diff.cpp.o.requires
	$(MAKE) -f CMakeFiles/nn_control_tester.dir/build.make CMakeFiles/nn_control_tester.dir/diff.cpp.o.provides.build
.PHONY : CMakeFiles/nn_control_tester.dir/diff.cpp.o.provides

CMakeFiles/nn_control_tester.dir/diff.cpp.o.provides.build: CMakeFiles/nn_control_tester.dir/diff.cpp.o


CMakeFiles/nn_control_tester.dir/matrix_vector_ops.cpp.o: CMakeFiles/nn_control_tester.dir/flags.make
CMakeFiles/nn_control_tester.dir/matrix_vector_ops.cpp.o: ../matrix_vector_ops.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/eric/Programs/Snapdragon/nn_control/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_5) "Building CXX object CMakeFiles/nn_control_tester.dir/matrix_vector_ops.cpp.o"
	/usr/bin/c++   $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/nn_control_tester.dir/matrix_vector_ops.cpp.o -c /home/eric/Programs/Snapdragon/nn_control/matrix_vector_ops.cpp

CMakeFiles/nn_control_tester.dir/matrix_vector_ops.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/nn_control_tester.dir/matrix_vector_ops.cpp.i"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/eric/Programs/Snapdragon/nn_control/matrix_vector_ops.cpp > CMakeFiles/nn_control_tester.dir/matrix_vector_ops.cpp.i

CMakeFiles/nn_control_tester.dir/matrix_vector_ops.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/nn_control_tester.dir/matrix_vector_ops.cpp.s"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/eric/Programs/Snapdragon/nn_control/matrix_vector_ops.cpp -o CMakeFiles/nn_control_tester.dir/matrix_vector_ops.cpp.s

CMakeFiles/nn_control_tester.dir/matrix_vector_ops.cpp.o.requires:

.PHONY : CMakeFiles/nn_control_tester.dir/matrix_vector_ops.cpp.o.requires

CMakeFiles/nn_control_tester.dir/matrix_vector_ops.cpp.o.provides: CMakeFiles/nn_control_tester.dir/matrix_vector_ops.cpp.o.requires
	$(MAKE) -f CMakeFiles/nn_control_tester.dir/build.make CMakeFiles/nn_control_tester.dir/matrix_vector_ops.cpp.o.provides.build
.PHONY : CMakeFiles/nn_control_tester.dir/matrix_vector_ops.cpp.o.provides

CMakeFiles/nn_control_tester.dir/matrix_vector_ops.cpp.o.provides.build: CMakeFiles/nn_control_tester.dir/matrix_vector_ops.cpp.o


# Object files for target nn_control_tester
nn_control_tester_OBJECTS = \
"CMakeFiles/nn_control_tester.dir/tester.cpp.o" \
"CMakeFiles/nn_control_tester.dir/data.cpp.o" \
"CMakeFiles/nn_control_tester.dir/testDynamics.cpp.o" \
"CMakeFiles/nn_control_tester.dir/diff.cpp.o" \
"CMakeFiles/nn_control_tester.dir/matrix_vector_ops.cpp.o"

# External object files for target nn_control_tester
nn_control_tester_EXTERNAL_OBJECTS =

nn_control_tester: CMakeFiles/nn_control_tester.dir/tester.cpp.o
nn_control_tester: CMakeFiles/nn_control_tester.dir/data.cpp.o
nn_control_tester: CMakeFiles/nn_control_tester.dir/testDynamics.cpp.o
nn_control_tester: CMakeFiles/nn_control_tester.dir/diff.cpp.o
nn_control_tester: CMakeFiles/nn_control_tester.dir/matrix_vector_ops.cpp.o
nn_control_tester: CMakeFiles/nn_control_tester.dir/build.make
nn_control_tester: CMakeFiles/nn_control_tester.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/eric/Programs/Snapdragon/nn_control/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_6) "Linking CXX executable nn_control_tester"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/nn_control_tester.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/nn_control_tester.dir/build: nn_control_tester

.PHONY : CMakeFiles/nn_control_tester.dir/build

CMakeFiles/nn_control_tester.dir/requires: CMakeFiles/nn_control_tester.dir/tester.cpp.o.requires
CMakeFiles/nn_control_tester.dir/requires: CMakeFiles/nn_control_tester.dir/data.cpp.o.requires
CMakeFiles/nn_control_tester.dir/requires: CMakeFiles/nn_control_tester.dir/testDynamics.cpp.o.requires
CMakeFiles/nn_control_tester.dir/requires: CMakeFiles/nn_control_tester.dir/diff.cpp.o.requires
CMakeFiles/nn_control_tester.dir/requires: CMakeFiles/nn_control_tester.dir/matrix_vector_ops.cpp.o.requires

.PHONY : CMakeFiles/nn_control_tester.dir/requires

CMakeFiles/nn_control_tester.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/nn_control_tester.dir/cmake_clean.cmake
.PHONY : CMakeFiles/nn_control_tester.dir/clean

CMakeFiles/nn_control_tester.dir/depend:
	cd /home/eric/Programs/Snapdragon/nn_control/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/eric/Programs/Snapdragon/nn_control /home/eric/Programs/Snapdragon/nn_control /home/eric/Programs/Snapdragon/nn_control/build /home/eric/Programs/Snapdragon/nn_control/build /home/eric/Programs/Snapdragon/nn_control/build/CMakeFiles/nn_control_tester.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/nn_control_tester.dir/depend


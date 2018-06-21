cmake_minimum_required(VERSION 2.8.11)
project(nn_control)

set(GCC_COVERAGE_COMPILE_FLAGS
  "-Wall -Wshadow -O3 -g -march=native -std=c++14 -D_BSD_SOURCE -Wno-write-strings"
)
set(GCC_COVERAGE_LINK_FLAGS "")

include_directories($ENV{MY_INCLUDES_PATH})

add_executable(nn_control_server
  trainer.cpp
  simulator.cpp
  data.cpp
  dynamics.cpp
  diff.cpp
  #genann.c
  matrix_vector_ops.cpp
)

add_executable(test_genann
  test_genann.c
  genann.c
)

add_executable(nn_control_tester
  tester.cpp
  data.cpp
  testDynamics.cpp
  diff.cpp
  matrix_vector_ops.cpp
)


set(CMAKE_CXX_FLAGS
  "${CMAKE_CXX_FLAGS} ${GCC_COVERAGE_COMPILE_FLAGS}"
)
set(CMAKE_EXE_LINKER_FLAGS
  "${CMAKE_EXE_LINKER_FLAGS} ${GCC_COVERAGE_LINK_FLAGS}"
)

set (LIBS4LINK
  m
  pthread
  gsl
  gslcblas
)
target_link_libraries(nn_control_server
  ${LIBS4LINK}
)
target_link_libraries(test_genann
  ${LIBS4LINK}
)
target_link_libraries(nn_control_tester
  ${LIBS4LINK}
)

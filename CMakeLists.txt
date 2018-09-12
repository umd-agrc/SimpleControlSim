cmake_minimum_required(VERSION 2.8.11)
project(nn_control)

############################ My setup 

set(GCC_COVERAGE_COMPILE_FLAGS
  "-Wall -Wshadow -O3 -g -march=native -D_BSD_SOURCE -Wno-write-strings -Wno-shadow\
   -Wno-delete-non-virtual-dtor -std=c++17"
)
set(GCC_COVERAGE_LINK_FLAGS "")

if (BUILD_DEBUG)
  set(GCC_COVERAGE_COMPILE_FLAGS 
    "${GCC_COVERAGE_COMPILE_FLAGS} -DDEBUG"
  )
else()
  message("${GCC_COVERAGE_COMPILE_FLAGS}")
endif()

set(HEADER_FILES
  $ENV{MY_INCLUDES_PATH}
)

option(BUILD_TESTS "Build tests" ON)
option(BUILD_TOOLS "Build tools" ON)
option(BUILD_MAIN "Build main" ON)

set (LIBS4LINK
  m
  pthread
  pthreadpool
  cpuinfo
  gsl
  lapack
  cblas
  f77blas
  atlas
  mxnet
)

if (BUILD_TESTS)
  add_subdirectory(test)
endif()

if(BUILD_TOOLS)
  add_subdirectory(tools)
endif()

if(BUILD_MAIN)
  add_executable(nn_control_server
    utils.cpp
    probability_distributions.cpp
    matrix_vector_ops.cpp
    data.cpp
    dynamics.cpp
    diff.cpp
    policy.cpp
    policy_eval.cpp
    runner.cpp
    simulator.cpp
    trainer.cpp
  )

  target_include_directories(nn_control_server PUBLIC
    ${HEADER_FILES}
  )

  target_link_libraries(nn_control_server
    ${LIBS4LINK}
    ${REQUIRED_LIBRARIES}
  )
endif()

set(CMAKE_CXX_FLAGS
  "${CMAKE_CXX_FLAGS} ${GCC_COVERAGE_COMPILE_FLAGS}"
)
set(CMAKE_EXE_LINKER_FLAGS
  "${CMAKE_EXE_LINKER_FLAGS} ${GCC_COVERAGE_LINK_FLAGS}"
)


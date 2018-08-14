cmake_minimum_required(VERSION 2.8.11)
project(nn_control)

set(CMAKE_MODULE_PATH
  ${CMAKE_MODULE_PATH}
  "/usr/local/share/TinyDNN"
)

############################ Tiny-dnn setup
option(USE_SSE        "Build tiny-dnn with SSE library support"     ON)
option(USE_AVX        "Build tiny-dnn with AVX library support"     ON)
option(USE_AVX2       "Build tiny-dnn with AVX2 library support"   OFF)
option(USE_TBB        "Build tiny-dnn with TBB library support"    OFF)
option(USE_OMP        "Build tiny-dnn with OMP library support"    OFF)
option(USE_NNPACK     "Build tiny-dnn with NNPACK library support" OFF)
option(USE_CBLAS      "Build tiny-dnn with CBLAS library support" ON)
#option(USE_OPENCL     "Build tiny-dnn with OpenCL library support" ON) 
option(USE_OPENCL     "Build tiny-dnn with OpenCL library support" OFF) 
option(USE_LIBDNN     "Build tiny-dnn with GreenteaLibDNN library support" OFF)
option(USE_SERIALIZER "Build tiny-dnn with Serialization support"   ON)
option(USE_DOUBLE     "Build tiny-dnn with double precision computations"  OFF)
option(USE_IMAGE_API  "Build tiny-dnn with Image API support"       ON)
option(USE_GEMMLOWP   "Build tiny-dnn with gemmlowp support"       OFF)


# Tiny-dnn provides a couple of multithreading solutions.
# The user can specify to use Intel Threading Building Blocks (TBB)
# or Open Multi-Processing (OpenMP) as a backend for multi threading
# processing. In case that none of this libraries are required, tiny-dnn
# will use the standard C++11 Thread support library.

# Find Intel Threading Building Blocks (TBB)
find_package(TBB QUIET)
if(USE_TBB AND TBB_FOUND)
    message(STATUS "Found Intel TBB: ${TBB_INCLUDE_DIR}")
    # In case that TBB is found we force to disable OpenMP since
    # tiny-dnn does not support mutiple multithreading backends.
    set(USE_OMP OFF)
    #TODO: add definitions in configure
    add_definitions(-DCNN_USE_TBB)
    include_directories(${TBB_INCLUDE_DIRS})
    link_directories(${TBB_LIBRARY_DIRS})
    list(APPEND REQUIRED_LIBRARIES ${TBB_LIBRARIES})
elseif(USE_TBB AND NOT TBB_FOUND)
    # In case the user sets the flag USE_TBB to ON, the CMake build-tree
    # will require to find TBB in your system. Otherwise, the user can
    # set the paths to headers and libs by hand.
    message(FATAL_ERROR "Intel TBB not found. Please set TBB_INCLUDE_DIRS & "
            "TBB_LIBRARIES")
endif()

if(NOT USE_SERIALIZER)
    add_definitions(-DCNN_NO_SERIALIZATION)
endif()

if(USE_DOUBLE)
    add_definitions(-DCNN_USE_DOUBLE)
endif()

if(USE_IMAGE_API)
    add_definitions(-DDNN_USE_IMAGE_API)
endif()

if(USE_GEMMLOWP)
    add_definitions(-DUSE_GEMMLOWP)
endif()

# Find Open Multi-Processing (OpenMP)
find_package(OpenMP QUIET)
if(USE_OMP AND OPENMP_FOUND)
    message(STATUS "Found OpenMP")
    # In case that OMP is found we force to disable Intel TBB since
    # tiny-dnn does not support mutiple multithreading backends.
    set(USE_TBB OFF)
    add_definitions(-DCNN_USE_OMP)
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
elseif(USE_OMP AND NOT OPENMP_FOUND)
    # In case the user sets the flag USE_OMP to ON, the CMake build-tree
    # will require to find OMP in your system. Otherwise, the user can
    # set the CMAKE_C_FLAGS and CMAKE_CXX_FLAGS by hand.
    message(FATAL_ERROR "Can't find OpenMP. Please set OpenMP_C_FLAGS & "
            "OpenMP_CXX_FLAGS")
endif()

# Find NNPACK: Acceleration package for neural networks on multi-core CPUs
find_package(NNPACK QUIET)
if(USE_NNPACK AND NNPACK_FOUND)
    message(STATUS "Found NNPACK: ${NNPACK_INCLUDE_DIR}")
    add_definitions(-DCNN_USE_NNPACK)
    include_directories(SYSTEM ${NNPACK_INCLUDE_DIR})
    include_directories(SYSTEM ${NNPACK_INCLUDE_DIR}/../third-party/pthreadpool/include)
    list(APPEND REQUIRED_LIBRARIES ${NNPACK_LIB})
elseif(USE_NNPACK AND NOT NNPACK_FOUND)
    # In case the user sets the flag USE_NNPACK to ON, the CMake build-tree
    # will require to find NNPACK in your system. Otherwise, the user can
    # set the paths to headers and libs by hand.
    message(FATAL_ERROR "Can't find NNPACK. Please set NNPACK_INCLUDE_DIR "
            " & NNPACK_LIB")
endif()

# Find CBLAS: Acceleration package for linear algebra
if(USE_CBLAS)
    # In case the user sets the flag USE_CBLAS to ON, the CMake build-tree
    # will require to find CBLAS in your system.
    find_package(BLAS REQUIRED)
    if(BLAS_FOUND)
      #if (NOT EXISTS "/usr/include/cblas.h" OR NOT EXISTS ${BLAS_LIBRARIES})
      #message(FATAL_ERROR "CBLAS path error.")
      #endif()
        add_definitions(-DCNN_USE_CBLAS)
        list(APPEND REQUIRED_LIBRARIES ${BLAS_LIBRARIES})
    else()
        message(FATAL_ERROR "Can't find CBLAS. On Ubunbu, you may want to "
                "install it by:\n  sudo apt-get install libatlas-dev")
    endif()
endif()

# in case that TBB and OMP are not enabled/found,
# we enable standard C++11 multithread support.
if((NOT USE_TBB) AND (NOT USE_OMP) AND (NOT WIN32))
    #list(APPEND EXTRA_C_FLAGS -pthread)
    set(USE_PTHREAD ON)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -pthread")
    message(STATUS "TBB and OMP disabled: Using Pthread instead.")
else((NOT USE_TBB) AND (NOT USE_OMP))
    set(USE_PTHREAD OFF)
endif((NOT USE_TBB) AND (NOT USE_OMP) AND (NOT WIN32))

find_package(OpenCL QUIET)
if(USE_OPENCL AND OpenCL_FOUND)
    message(STATUS "Found OpenCL: ${OpenCL_INCLUDE_DIRS}")
    #add_definitions(-DCNN_HAVE_OPENCL)
    add_definitions(-DUSE_OPENCL)
    include_directories(SYSTEM ${OpenCL_INCLUDE_DIRS})
    list(APPEND REQUIRED_LIBRARIES ${OpenCL_LIBRARY})
elseif(USE_OPENCL AND NOT OpenCL_FOUND)
    # In case the user sets the flag USE_OPENCL to ON, the CMake build-tree
    # will require to find OPENCL in your system. Otherwise, the user can
    # set the paths to headers and libs by hand.
    message(FATAL_ERROR "Can't find OpenCL.")
endif()

find_package(GreenteaLibDNN QUIET)
if(OpenCL_FOUND AND USE_LIBDNN AND GreenteaLibDNN_FOUND)
    message(STATUS "Found GreenteaLibDNN: ${GREENTEA_INCLUDE_DIRS}")
    add_definitions(-DCNN_USE_LIBDNN)
    include_directories(SYSTEM ${GREENTEA_INCLUDE_DIRS})
    list(APPEND REQUIRED_LIBRARIES greentea_libdnn ${GREENTEA_LIBRARIES})
elseif(USE_LIBDNN AND NOT OpenCL_FOUND)
    message(FATAL_ERROR "OpenCL is needed for GreenteaLibDNN.")
elseif(USE_LIBDNN AND NOT LIBDNN_FOUND)
    # In case the user sets the flag USE_LIBDNN to ON, the CMake build-tree
    # will require to find LibDNN in your system. Otherwise, the user can
    # set the paths to headers and libs by hand.
    message(FATAL_ERROR "Can't find LibDNN.")
endif()

####
# Setup the compiler options

# set c++ standard to c++14.
# Note: not working on CMake 2.8. We assume that user has
#       a compiler with C++14 support.

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
message(STATUS "C++14 support has been enabled by default.")

# Unix
if(CMAKE_COMPILER_IS_GNUCXX OR MINGW OR
   CMAKE_CXX_COMPILER_ID MATCHES "Clang")
    include(CheckCXXCompilerFlag)
    check_cxx_compiler_flag("-msse3" COMPILER_HAS_SSE_FLAG)
    check_cxx_compiler_flag("-mavx"  COMPILER_HAS_AVX_FLAG)
    check_cxx_compiler_flag("-mavx2" COMPILER_HAS_AVX2_FLAG)
    check_cxx_compiler_flag("-mfma" COMPILER_HAS_AVX2_FLAG)

    # set Streaming SIMD Extension (SSE) instructions
    if(USE_SSE AND COMPILER_HAS_SSE_FLAG)
        add_definitions(-DCNN_USE_SSE)
        set(EXTRA_C_FLAGS "${EXTRA_C_FLAGS} -msse3")
    endif(USE_SSE AND COMPILER_HAS_SSE_FLAG)
    # set Advanced Vector Extensions (AVX)
    if(USE_AVX AND COMPILER_HAS_AVX_FLAG)
        add_definitions(-DCNN_USE_AVX)
        set(EXTRA_C_FLAGS "${EXTRA_C_FLAGS} -mavx")
    endif(USE_AVX AND COMPILER_HAS_AVX_FLAG)
    # set Advanced Vector Extensions 2 (AVX2)
    if(USE_AVX2 AND COMPILER_HAS_AVX2_FLAG)
        add_definitions(-DCNN_USE_AVX2)
        set(EXTRA_C_FLAGS "${EXTRA_C_FLAGS} -mavx2 -mfma -march=core-avx2")
    endif(USE_AVX2 AND COMPILER_HAS_AVX2_FLAG)

    # include extra flags to the compiler
    # TODO: add info about those flags.
    set(EXTRA_C_FLAGS "${EXTRA_C_FLAGS} -Wall -Wpedantic -Wno-narrowing -Wno-deprecated")
    set(EXTRA_C_FLAGS_RELEASE "${EXTRA_C_FLAGS_RELEASE} -O3")
    set(EXTRA_C_FLAGS_DEBUG   "${EXTRA_C_FLAGS_DEBUG} -g3 -pthread")
elseif(MSVC)
    if(USE_SSE)
        add_definitions(-DCNN_USE_SSE)
        set(EXTRA_C_FLAGS "${EXTRA_C_FLAGS} /arch:SSE2")
    endif(USE_SSE)
    if(USE_AVX)
        add_definitions(-DCNN_USE_AVX)
        set(EXTRA_C_FLAGS "${EXTRA_C_FLAGS} /arch:AVX")
    endif(USE_AVX)
    if(USE_AVX2)
        add_definitions(-DCNN_USE_AVX2)
        set(EXTRA_C_FLAGS "${EXTRA_C_FLAGS} /arch:AVX2")
    endif(USE_AVX2)
    # include specific flags for release and debug modes.
    set(EXTRA_C_FLAGS_RELEASE "${EXTRA_C_FLAGS_RELEASE}
        /Ox /Oi /Ot /Oy /GL /fp:fast /GS-")
    set(CMAKE_EXE_LINKER_FLAGS_RELEASE "${CMAKE_EXE_LINKER_FLAGS_RELEASE} /LTCG")
    set(EXTRA_C_FLAGS_DEBUG "${EXTRA_C_FLAGS_DEBUG}")
    set(EXTRA_C_FLAGS "${EXTRA_C_FLAGS} /W4 /bigobj")
    # this is fine
    add_definitions(-D _CRT_SECURE_NO_WARNINGS)
    add_definitions(-D _SCL_SECURE_NO_WARNINGS)
    # prolly powerless with header-only project
    set(EXTRA_C_FLAGS "${EXTRA_C_FLAGS} /MP")
endif()

####
# Set compiler options
set(CMAKE_CXX_FLAGS         "${CMAKE_CXX_FLAGS} ${EXTRA_C_FLAGS}")
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} ${EXTRA_C_FLAGS_RELEASE}")
set(CMAKE_CXX_FLAGS_DEBUG   "${CMAKE_CXX_FLAGS_DEBUG} ${EXTRA_C_FLAGS_DEBUG}")

if(PROFILE AND (CMAKE_COMPILER_IS_GNUCXX OR MINGW OR
    CMAKE_CXX_COMPILER_ID MATCHES "Clang"))
    set(CMAKE_CXX_FLAGS     "${CMAKE_CXX_FLAGS} -pg")
endif()

if (USE_ASAN)
    #enable ASan
    set(ENV{ASAN_OPTIONS} "strict_string_checks=1:detect_stack_use_after_return=1:check_initialization_order=1:strict_init_order=1:symbolize=1")
    CHECK_CXX_COMPILER_FLAG(-fsanitize-address-use-after-scope HasUseAfterScope)
    set(ASAN_FLAGS "-fsanitize=address -fno-omit-frame-pointer")
    if (HasUseAfterScope)
        set(ASAN_FLAGS "${ASAN_FLAGS} -fsanitize-address-use-after-scope")
    endif()
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${ASAN_FLAGS}")
endif()

############################ My setup 

set(GCC_COVERAGE_COMPILE_FLAGS
  "-Wall -Wshadow -O3 -g -march=native -D_BSD_SOURCE -Wno-write-strings -Wno-shadow\
   -Wno-delete-non-virtual-dtor -std=c++17"
)
set(GCC_COVERAGE_LINK_FLAGS "")

set(HEADER_FILES
  $ENV{MY_INCLUDES_PATH}
)

option(BUILD_TESTS "Build tests" ON)
option(BUILD_MAIN "Build tests" ON)
add_definitions(-DNDEBUG)

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
)

if (BUILD_TESTS)
  add_subdirectory(test)
endif()

if(BUILD_MAIN)
  add_executable(nn_control_server
    trainer.cpp
    simulator.cpp
    data.cpp
    dynamics.cpp
    diff.cpp
    matrix_vector_ops.cpp
    policy.cpp
    runner.cpp
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


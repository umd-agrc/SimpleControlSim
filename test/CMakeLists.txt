add_executable(test_mxnet
  testMxNet.cpp
  ../utils.cpp
  ../policy.cpp
  ../policy_eval.cpp
  ../probability_distributions.cpp
  ../runner.cpp
  ../diff.cpp
  ../dynamics.cpp
  ../data.cpp
)

target_include_directories(test_mxnet PUBLIC
  ${PROJECT_SOURCE_DIR} 
  ${HEADER_FILES}
)

target_link_libraries(test_mxnet
  ${LIBS4LINK}
)

add_executable(test_sim
  testSim.cpp
)

target_include_directories(test_sim PUBLIC
  ${PROJECT_SOURCE_DIR} 
  ${HEADER_FILES}
)

target_link_libraries(test_sim
  ${LIBS4LINK}
)

set(CMAKE_CXX_FLAGS
  "${CMAKE_CXX_FLAGS} ${GCC_COVERAGE_COMPILE_FLAGS}"
)
set(CMAKE_EXE_LINKER_FLAGS
  "${CMAKE_EXE_LINKER_FLAGS} ${GCC_COVERAGE_LINK_FLAGS}"
)

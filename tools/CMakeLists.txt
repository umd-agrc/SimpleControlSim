add_executable(data_converter
  data_converter.cpp
)

target_include_directories(data_converter PUBLIC
  ${PROJECT_SOURCE_DIR} 
  ${HEADER_FILES}
)

target_link_libraries(data_converter
  ${LIBS4LINK}
)

set(CMAKE_CXX_FLAGS
  "${CMAKE_CXX_FLAGS} ${GCC_COVERAGE_COMPILE_FLAGS}"
)
set(CMAKE_EXE_LINKER_FLAGS
  "${CMAKE_EXE_LINKER_FLAGS} ${GCC_COVERAGE_LINK_FLAGS}"
)

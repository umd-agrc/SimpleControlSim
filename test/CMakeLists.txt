add_executable(test_mxnet
  testMxNet.cpp
)

target_include_directories(test_mxnet PUBLIC
  ${PROJECT_SOURCE_DIR} 
  ${HEADER_FILES}
)

target_link_libraries(test_mxnet
  ${LIBS4LINK}
)

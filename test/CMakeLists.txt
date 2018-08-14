add_executable(test_cppad
  testCppAd.cpp
)

target_include_directories(test_cppad PUBLIC
  ${PROJECT_SOURCE_DIR} 
  ${HEADER_FILES}
)

target_link_libraries(test_cppad
  ${LIBS4LINK}
)

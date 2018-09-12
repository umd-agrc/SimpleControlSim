#ifndef NN_CTL_TEST_MXNET_H_
#define NN_CTL_TEST_MXNET_H_

#include <iostream>
#include <vector>
#include <chrono>
#include <mxnet-cpp/MxNetCpp.h>

#include "utils.h"
#include "test_defines.h"
#include "policy.h"
#include "policy_eval.h"

int testGrad();
int testNetUpdate();
void testNetReshape();
int testNDArrayBasics();
int testSimpleArithmetic();
int testNDArrayReduce();
int testExp();
int testNDArrayAssign();
int testNDArrayConcat();
int testNDArrayInit();
int testNDArrayMax();
int testNDArrayCopy();
void testNDArrayIo();
int testSymbolReduce();
int testSymbolIo();
int testLogging();

#endif

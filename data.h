#pragma once

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <iostream>
#include <map>
#include <vector>
#include <mxnet-cpp/MxNetCpp.h>

#include "dynamics.h"
#include "defines.h"

int seekChar(const char *s, int c);
int seekCharSequence(const char *s, const char *c, int *idx);
void printDoubleArray(double *arr, int len, char *name);
void logNDArrayMap(std::string fileprefix,
                   std::string filesuffix,
                   std::map<std::string,mxnet::cpp::NDArray> m);

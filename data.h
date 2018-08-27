#pragma once

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "dynamics.h"
#include "defines.h"

#define APPEND_LINE "a"
#define NEW_LINE "n"

int seekChar(const char *s, int c);
int seekCharSequence(const char *s, const char *c, int *idx);
int load_data(const char *training_data,
              int *datalen,
              mxnet::cpp::NDArray *input,
              mxnet::cpp::NDArray *output,
              int *samples);
int save_data(const char *training_data,
              const char *header,
              mxnet::cpp::NDArray *input,
              mxnet::cpp::NDArray *output);

// Log vector to file
// If specified with mode "a", then append line (no newline)
// If specified with mode "w", then add newline at the end
// Can specify to add delimiter to the end of the vector
int logVector(FILE *logfile, std::vector<mx_float> *v, char *dlm, char *mode, bool endDlm);
int logHeader(FILE *logfile,char *header);
//TODO log time
int logTime(FILE *logfile,double t, char *dlm, char *mode);
void printDoubleArray(double *arr, int len, char *name);

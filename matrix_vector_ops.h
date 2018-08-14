#pragma once

#include <math.h>

#include "defines.h"

std::vector<double> vector_stack(const std::vector<double> *a, const std::vector<double> *b);
std::vector<double> vector_sub(const std::vector<double> *a, const std::vector<double> *b);
std::vector<double> vector_add(const std::vector<double> *a, const std::vector<double> *b);
std::vector<double> vector_add(const std::vector<double> *a, double b);
std::vector<double> vector_scale(const std::vector<double> *a, double k);
double vector_infnorm(const std::vector<double> *v);
void vector_print(const std::vector<double> *v, char *name);
std::vector<double> vector_reduce_sum(const std::vector<double> *a);
std::vector<double> DataPoint_reduce_sum(const std::vector<DataPoint> *a);
std::vector<double> DataPoint_mean_square(const std::vector<DataPoint> *a,
                                          std::vector<double> *mean);
std::vector<double> vector_create(double d,int size);
std::vector<double> vector_edivide(const std::vector<double> *a,
                                   const std::vector<double> *b);

#pragma once

#include <math.h>

#include "defines.h"

std::vector<mx_float> vector_stack(
    const std::vector<mx_float> *a, const std::vector<mx_float> *b);
std::vector<mx_float> vector_sub(
    const std::vector<mx_float> *a, const std::vector<mx_float> *b);
std::vector<mx_float> vector_add(
    const std::vector<mx_float> *a, const std::vector<mx_float> *b);
std::vector<mx_float> vector_add(
    const std::vector<mx_float> *a, mx_float b);
std::vector<mx_float> vector_scale(
    const std::vector<mx_float> *a, mx_float k);
mx_float vector_infnorm(
    const std::vector<mx_float> *v);
void vector_print(
    const std::vector<mx_float> *v, char *name);
std::vector<mx_float> vector_reduce_sum(
    const std::vector<mx_float> *a);
std::vector<mx_float> DataPoint_reduce_sum(
    const std::vector<DataPoint> *a);
std::vector<mx_float> DataPoint_mean_square(
    const std::vector<DataPoint> *a,
    std::vector<mx_float> *mean);
std::vector<mx_float> vector_create(
    mx_float d,int size);
std::vector<mx_float> vector_edivide(
    const std::vector<mx_float> *a,
    const std::vector<mx_float> *b);

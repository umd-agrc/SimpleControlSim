#pragma once

#include <assert.h>
#include <math.h>
#include <mxnet-cpp/MxNetCpp.h>

#include "defines.h"
#include "utils.h"
#include "matrix_vector_ops.h"

#define DEFAULT_DIFF_ERROR_TOLERANCE 1e-5
#define DEFAULT_DIFF_STEPSIZE 1e-2

#define MAX_DIFF_CORRECTION_ATTEMPTS 100

#define RUNGE_KUTTA_ORDER 5

typedef int (*DynamicsFunction) (mxnet::cpp::NDArray *dy,
                                 mx_float t,
                                 mxnet::cpp::NDArray *y,
                                 mxnet::cpp::NDArray *u);

// Runge-Kutte-Dormand-Prince embedded method modified for controlled systems
//TODO implement adaptive step size
int rungeKutteStep(DynamicsFunction dyn,
                   mx_float t,
                   mxnet::cpp::NDArray *y_next,
                   mxnet::cpp::NDArray *rk_e_next,
                   VehicleState *vehicle,
                   Controller *controller,
                   mx_float h);

int rungeKutteAdaptiveStep(DynamicsFunction dyn,
                           mx_float t,
                           mxnet::cpp::NDArray *y_next,
                           mxnet::cpp::NDArray *rk_e_next,
                           VehicleState *vehicle,
                           Controller *controller,
                           mx_float *stepSize,
                           mx_float tolerance,
                           bool reset);

//TODO implement implicit RK method as well for stiff equations

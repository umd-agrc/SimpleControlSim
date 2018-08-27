#pragma once

#include <assert.h>
#include <math.h>
//#include <gsl/gsl_matrix.h>
//#include <gsl/gsl_vector.h>

#include "defines.h"
#include "matrix_vector_ops.h"

#define DEFAULT_DIFF_ERROR_TOLERANCE 1e-5
#define DEFAULT_DIFF_STEPSIZE 1e-2

#define MAX_DIFF_CORRECTION_ATTEMPTS 100

#define RUNGE_KUTTA_ORDER 5

typedef int (*DynamicsFunction) (std::vector<mx_float> *dy,
                                 mx_float t,
                                 const std::vector<mx_float> *y,
                                 const std::vector<mx_float> *u);

// Runge-Kutte-Dormand-Prince embedded method modified for controlled systems
//TODO implement adaptive step size
int rungeKutteStep(DynamicsFunction dyn,
                   mx_float t,
                   std::vector<mx_float> *y_next,
                   std::vector<mx_float> *rk_e_next,
                   const VehicleState *vehicle,
                   const Controller *controller,
                   mx_float h);

int rungeKutteAdaptiveStep(DynamicsFunction dyn,
                           mx_float t,
                           std::vector<mx_float> *y_next,
                           std::vector<mx_float> *rk_e_next,
                           const VehicleState *vehicle,
                           const Controller *controller,
                           mx_float *stepSize,
                           mx_float tolerance,
                           bool reset);

//TODO implement implicit RK method as well for stiff equations

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

/*
typedef int (*DynamicsFunction) (gsl_vector *dy,
                                 double t,
                                 const gsl_vector *y,
                                 const gsl_vector *u);
                                 */

typedef int (*DynamicsFunction) (std::vector<double> *dy,
                                 double t,
                                 const std::vector<double> *y,
                                 const std::vector<double> *u);

// Runge-Kutte-Dormand-Prince embedded method modified for controlled systems
//TODO implement adaptive step size
int rungeKutteStep(DynamicsFunction dyn,
                   double t,
                   std::vector<double> *y_next,
                   std::vector<double> *rk_e_next,
                   const VehicleState *vehicle,
                   const Controller *controller,
                   double h);

int rungeKutteAdaptiveStep(DynamicsFunction dyn,
                           double t,
                           std::vector<double> *y_next,
                           std::vector<double> *rk_e_next,
                           const VehicleState *vehicle,
                           const Controller *controller,
                           double *stepSize,
                           double tolerance,
                           bool reset);

//TODO implement implicit RK method as well for stiff equations

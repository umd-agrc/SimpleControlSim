#pragma once

#include <gsl/gsl_matrix.h>
#include <gsl/gsl_vector.h>

#include "defines.h"
#include "matrix_vector_ops.h"

#define NUM_STATES 1
#define NUM_INPUTS 0

// A-stability test of RK method:
// y' = ky
int testDynamics(gsl_vector *dy, double t, const gsl_vector *y, const gsl_vector *u);

int setupTestDynamics();
int teardownTestDynamics();

gsl_vector *testFeedback(gsl_vector *yd, gsl_vector *y);

int setupTestFeedback();
int teardownTestFeedback();

#include <math.h>
#include <gsl/gsl_blas.h>

#include "testDynamics.h"

gsl_matrix *dyn;
gsl_vector *fb;

int testDynamics(gsl_vector *dy, double t, const gsl_vector *y, const gsl_vector *u) {
  gsl_blas_dgemv(CblasNoTrans,
                 1.0, dyn, y, 
                 0.0, dy);

  return SIM_SUCCESS;
}

int setupTestDynamics() {
  dyn = gsl_matrix_calloc(NUM_STATES,NUM_STATES+NUM_INPUTS);

  // For stability, k < 0
  double k = -1;
  gsl_matrix_set(dyn, 0, 0, k);

  return SIM_SUCCESS;
}

int teardownTestDynamics() {
  gsl_matrix_free(dyn);
  return SIM_SUCCESS;
}

// Return unallocated feedback matrix
gsl_vector *testFeedback(gsl_vector *yd, gsl_vector *y) {
  return fb;
}

// Nothing. No need to allocate memory
int setupTestFeedback() {
  return SIM_SUCCESS;
}

// Nothing. No need to deallocate memory
int teardownTestFeedback() {
  return SIM_SUCCESS;
}

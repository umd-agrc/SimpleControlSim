#pragma once

#include <math.h>
#include <gsl/gsl_matrix.h>
#include <gsl/gsl_vector.h>

#include "defines.h"

// Pass matrices a, b
// Matrix c is already allocated and is the correct size
// c = [a b]'
int gsl_matrix_vstack(gsl_matrix *c, const gsl_matrix *a, const gsl_matrix *b);

// Multiply matrices c = a*b
// Use gsl_blas_dgemm() instead:
//  gsl_blas_dgemm(CblasNoTrans, CblasNoTrans, 1.0, A, B, 0.0, C);
int gsl_matrix_mul(gsl_matrix *c, const gsl_matrix *a, const gsl_matrix *b);

// Helper function for matrix multiplication to multiply specified row by
// specified column
// Number of columns in a must be equal to number of rows in b
double gsl_matrix_rowcol_mul(const gsl_matrix *a, const gsl_matrix *b, int r, int c);

// Pass vectors a, b
// Vector c is already allocated and is the correct size
// c = [a b]'
int gsl_vector_vstack(gsl_vector *c, const gsl_vector *a, const gsl_vector *b);

double gsl_vector_infnorm(const gsl_vector *v);

void gsl_vector_print(const gsl_vector *v, char *name);

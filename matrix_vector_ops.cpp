#include "matrix_vector_ops.h"

int gsl_matrix_vstack(gsl_matrix *c, const gsl_matrix *a, const gsl_matrix *b) {
  if (a->size2 != b->size2) {
    SIM_ERROR("Columns must be equal to do vertical matrix stack: %zd != %zd",
        a->size2, b->size2);
    return SIM_FAILURE;  
  }

  for (unsigned int i=0; i < a->size1; i++) {
   for (unsigned int j=0; j < a->size2; j++) {
      gsl_matrix_set(c, i, j, gsl_matrix_get(a, i, j));
    }
  }

  for (unsigned int i=0; i < b->size1; i++) {
    for (unsigned int j=0; j < b->size2; j++) {
      gsl_matrix_set(c, i + a->size1, j + a->size2, gsl_matrix_get(b, i, j));
    }
  }

  return SIM_SUCCESS;
}

// Use gsl_blas_dgemm() instead:
//  gsl_blas_dgemm(CblasNoTrans, CblasNoTrans, 1.0, A, B, 0.0, C);
int gsl_matrix_mul(gsl_matrix *c, const gsl_matrix *a, const gsl_matrix *b) {
  for (unsigned int i=0; i < c->size1; i++) {
    for (unsigned int j=0; j < c->size2; j++) {
      gsl_matrix_set(c, i, j, gsl_matrix_rowcol_mul(a,b,i,j));
    }
  }
  
  return SIM_SUCCESS;
}

double gsl_matrix_rowcol_mul(const gsl_matrix *a, const gsl_matrix *b, int r, int c) {
  double ret=0;
  for (unsigned int i=0; i < a->size2; i++) {
    ret += gsl_matrix_get(a, r, i)*gsl_matrix_get(b, i, c);
  }
  
  return ret;
}

int gsl_vector_vstack(gsl_vector *c, const gsl_vector *a, const gsl_vector *b) {
  for (unsigned int i=0; i < a->size; i++) {
      gsl_vector_set(c, i, gsl_vector_get(a, i));
  }

  for (unsigned int i=0; i < b->size; i++) {
      gsl_vector_set(c, i + a->size, gsl_vector_get(b, i));
  }

  return SIM_SUCCESS;
}

double gsl_vector_infnorm(const gsl_vector *v){ 
  double min,max;

  gsl_vector_minmax(v,&min,&max);
  return fabs(min) > fabs(max) ? fabs(min) : fabs(max);  
}

void gsl_vector_print(const gsl_vector *v,char *name) {
  printf("%s: ",name);
  for (unsigned int i = 0; i < v->size; i++) {
    printf("%g ", gsl_vector_get(v,i));
  }
  printf("\n");
}

#pragma once

#include <stdio.h>
#include <stdbool.h>
#include <gsl/gsl_matrix.h>
#include <gsl/gsl_vector.h>

#define SIM_SUCCESS 0
#define SIM_FAILURE -1

#define SIM_INFO(...) do {fprintf(stdout,##__VA_ARGS__);} while(0)
#define SIM_ERROR(...) do {fprintf(stdout,##__VA_ARGS__);} while(0)

#define SHUTDOWN_MESSAGE "shutdown\n"

typedef struct {
  gsl_vector *yd;
  gsl_vector *y;
} VehicleState;

typedef struct {
  gsl_vector *(*feedback) (gsl_vector *yd, gsl_vector *y);
} Controller;

#include "diff.h"

#include <assert.h>

//TODO test rk method for A-stability
//TODO rk runner to do adaptive step size stuff
int rungeKutteStep(DynamicsFunction dyn,
                   double t,
                   gsl_vector *y_next,
                   gsl_vector *rk_e_next,
                   const VehicleState *vehicle,
                   const Controller *controller,
                   double h) {
  static double c[7] = { 0, (double)1/5, (double)3/10,
                         (double)4/5, (double)8/9, (double)1, (double)1 };
  static double b[2][7] = 
        { { (double)35/384, 0, (double)500/1113, (double)125/192,
            (double)-2187/6784, (double)11/84, 0 },
          { (double)5179/57600, 0, (double)7571/16695, (double)393/640,
            (double)-92097/339200, (double)187/2100, (double)1/40 } };
  static double a[7][7] = 
        { { 0, 0, 0, 0, 0, 0, 0 },
          { (double)1/5, 0, 0, 0, 0, 0, 0 },
          { (double)3/40, (double)9/40, 0, 0, 0, 0, 0 },
          { (double)44/45, (double)-56/15, (double)32/9, 0, 0, 0, 0 },
          { (double)19372/6561, (double)-25360/2187, (double)65558/6561,
            (double)-212/729, 0, 0, 0 },
          { (double)9017/3168, (double)-355/33, (double)46732/5247,
            (double)49/176, (double)-5103/18656, 0, 0 },
          { (double)35/384, 0, (double)500/1113, (double)125/192,
            (double)-2187/6784, (double)11/84, 0 } };

  gsl_vector *k[7];
  for (int i=0; i < 7; i++) {
    k[i] = gsl_vector_calloc(vehicle->y->size);
  }
  gsl_vector *y_tmp = gsl_vector_calloc(vehicle->y->size);
  gsl_vector *k_tmp = gsl_vector_calloc(k[0]->size);

  // Evaluate R-K `k` components ---------------------
  for (int i=0; i < 7; i++) {
    gsl_vector_memcpy(y_tmp,vehicle->y);
    for (int j=0; j < i; j++) {
      gsl_vector_memcpy(k_tmp,k[j]);
      gsl_vector_scale(k_tmp,h*a[i][j]);
      gsl_vector_add(y_tmp,k_tmp);
    }
    assert(dyn(k[i],t+c[i]*h, y_tmp, controller->feedback(vehicle->yd,y_tmp)) ==
           SIM_SUCCESS);
  }
  // -------------------------------------------------

  gsl_vector *y_check = gsl_vector_calloc(vehicle->y->size);

  gsl_vector_memcpy(y_next,vehicle->y); 
  gsl_vector_memcpy(y_check,vehicle->y); 

  for (int i=0; i < 7; i++) {
    gsl_vector_memcpy(k_tmp,k[i]);
    gsl_vector_scale(k_tmp,h*b[0][i]);
    gsl_vector_add(y_next,k_tmp);

    gsl_vector_memcpy(k_tmp,k[i]);
    gsl_vector_scale(k_tmp,h*b[1][i]);
    gsl_vector_add(y_check,k_tmp);
  }
  gsl_vector_memcpy(rk_e_next,y_next); 
  gsl_vector_sub(rk_e_next,y_check);

  return SIM_SUCCESS;
}

int rungeKutteAdaptiveStep(DynamicsFunction dyn,
                           double t,
                           gsl_vector *y_next,
                           gsl_vector *rk_e_next,
                           const VehicleState *vehicle,
                           const Controller *controller,
                           double *stepSize,
                           double tolerance) {
  // Save previous step size as default
  static double h = DEFAULT_DIFF_STEPSIZE;
  static double stepUpdatePercentage = 0.9;

  if (tolerance < 0) tolerance = DEFAULT_DIFF_ERROR_TOLERANCE;

  rungeKutteStep(dyn,t,y_next,rk_e_next,vehicle,controller,h);
  double e = gsl_vector_infnorm(rk_e_next);
  int i=0;
  while (e > tolerance && i < MAX_DIFF_CORRECTION_ATTEMPTS) {
    // Update step size if error is too large
    // h = gamma*h*(tau/e)^(1/p)
    h = (h*pow((tolerance/e),1/RUNGE_KUTTA_ORDER))*stepUpdatePercentage;
    rungeKutteStep(dyn,t,y_next,rk_e_next,vehicle,controller,h);
    e = gsl_vector_infnorm(rk_e_next);
    i++;
  }
  if (e < tolerance) {
    h = (h*pow((tolerance/e),1/RUNGE_KUTTA_ORDER))/stepUpdatePercentage;
  }

  *stepSize = h;

  return SIM_SUCCESS;
}

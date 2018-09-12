#include "diff.h"

#include <assert.h>

using namespace mxnet::cpp;

//TODO test rk method for A-stability
//TODO rk runner to do adaptive step size stuff
int rungeKutteStep(DynamicsFunction dyn,
                   mx_float t,
                   NDArray *y_next,
                   NDArray *rk_e_next,
                   VehicleState *vehicle,
                   Controller *controller,
                   mx_float h) {
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
  std::vector<NDArray> k(7);
  NDArray y_tmp,k_tmp,k_res,u_tmp;
  // Evaluate R-K `k` components ---------------------
  for (int i=0; i < 7; i++) {
    k[i] = zeros(Shape(1,NUM_STATES));
    y_tmp = vehicle->y;
    for (int j=0; j < i; j++) {
      //k_tmp = k[j]*h*a[i][j];
      //y_tmp += k_tmp;
      y_tmp += k[j]*h*a[i][j];
    }
    u_tmp = *(controller->feedback(vehicle->yd,y_tmp,NULL,NULL));
    assert(dyn(&k_res,t+c[i]*h, &y_tmp, &u_tmp) ==
           SIM_SUCCESS);
    k[i] = k_res;
  }
  // -------------------------------------------------

  *y_next = vehicle->y;
  NDArray y_check = vehicle->y;

  for (int i=0; i < 7; i++) {
    *y_next += k[i]*h*b[0][i];
    y_check += k[i]*h*b[1][i];
  }
  *rk_e_next = *y_next - y_check;

  return SIM_SUCCESS;
}

int rungeKutteAdaptiveStep(DynamicsFunction dyn,
                           mx_float t,
                           NDArray *y_next,
                           NDArray *rk_e_next,
                           VehicleState *vehicle,
                           Controller *controller,
                           mx_float *stepSize,
                           mx_float tolerance,
                           bool reset) {
  // Save previous step size as default
  static mx_float h = DEFAULT_DIFF_STEPSIZE;
  static mx_float stepUpdatePercentage = 0.9;

  if (reset) h = DEFAULT_DIFF_STEPSIZE;
  if (tolerance < 0) tolerance = DEFAULT_DIFF_ERROR_TOLERANCE;

  rungeKutteStep(dyn,t,y_next,rk_e_next,vehicle,controller,h);
  mx_float e = max(*rk_e_next).At(0,0);
  int i=0;
  while (e > tolerance && i < MAX_DIFF_CORRECTION_ATTEMPTS) {
    // Update step size if error is too large
    // h = gamma*h*(tau/e)^(1/p)
    h = (h*pow((tolerance/e),1/RUNGE_KUTTA_ORDER))*stepUpdatePercentage;
    rungeKutteStep(dyn,t,y_next,rk_e_next,vehicle,controller,h);
    e = max(*rk_e_next).At(0,0);
    i++;
  }
  if (e < tolerance) {
    h = (h*pow((tolerance/e),1/RUNGE_KUTTA_ORDER))/stepUpdatePercentage;
  }

  *stepSize = h;

  return SIM_SUCCESS;
}

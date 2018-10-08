#include <math.h>

#include "dynamics.h"

#ifdef DEBUG
#include<chrono>
using namespace std::chrono;
static high_resolution_clock::time_point t1,t2;
#endif

using namespace mxnet::cpp;

NDArray dyn, gains;
NDArray fb;

PolicyFunction *policy;

int setupDynamics() {
  log("Setting up dynamics");

  mx_float X_u   =  -0.27996;
  mx_float Y_v   =  -0.22566;
  mx_float Z_w   =  -1.2991;
  mx_float L_p   =  -2.5110;
  mx_float M_q   =  -2.4467;
  mx_float N_r   =  -0.4948;
  mx_float X_th  =  -10.067;
  mx_float Y_ph  =   9.8648;
  mx_float L_ph  =  -21.358;
  mx_float M_th  =  -18.664;
  mx_float ph_p  =   0.9655;
  mx_float th_q  =   0.9634;
  mx_float ps_r  =   0.6748;
  mx_float Z_thr =  -39.282;
  mx_float L_la  =   11.468;
  mx_float M_lo  =   9.5711;
  mx_float N_ya  =   3.5647;
  mx_float ph_la =   0.0744;
  mx_float th_lo =   0.0594;
  mx_float ps_ya =   0.0397;

  std::vector<mx_float> dynVec = {
    0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, X_u, 0, 0, 0, 0, 0, 0, X_th, 0, 0, 0, 0, 0,
    0, 0, 0, 0, Y_v, 0, 0, 0, 0, Y_ph, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, Z_w, 0, 0, 0, 0, 0, 0, 0, 0, 0, Z_thr,
    0, 0, 0, 0, 0, 0, L_p, 0, 0, L_ph, 0, 0, L_la, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, M_q, 0, 0, M_th, 0, 0, M_lo, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, N_r, 0, 0, 0, 0, 0, N_ya, 0,
    0, 0, 0, 0, 0, 0, ph_p, 0, 0, 0, 0, 0, ph_la, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, th_q, 0, 0, 0, 0, 0, th_lo, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, ps_r, 0, 0, 0, 0, 0, ps_ya, 0,
  };
  dyn = NDArray(dynVec,
                Shape(NUM_STATES,NUM_STATES+NUM_INPUTS),
                Context::cpu());
  NDArray::WaitAll();

  return SIM_SUCCESS;
}

// Dynamics model from G. Gremillion, S. Humbert paper "System Identification of
// a Quadrotor Micro Air Vehicle"
// Augmented to include position dynamics
//
int dynamics(NDArray *dy, mx_float t, NDArray *y,
    NDArray *u) {
  (void) (t); // Avoid unused parameter warning
  /*
  NDArray::WaitAll();
  mx_float th = y->GetData()[11];
  //std::cout << "y " << *y << std::endl;
  dyn.SetData(0,3,cos(th));
  dyn.SetData(0,4,-sin(th));
  dyn.SetData(1,3,sin(th));
  dyn.SetData(1,4,cos(th));
  */
  //std::cout << "u " << *u << std::endl;
  NDArray::WaitAll();

  NDArray yu = Concat(*y,*u, Shape(NUM_INPUTS+NUM_STATES,1));

  //std::cout << "yu " << yu << std::endl;
  NDArray::WaitAll();
  *dy = dot(dyn,yu);
  //std::cout << "dy " << *dy << std::endl;
  NDArray::WaitAll();

  return SIM_SUCCESS;
}

NDArray *feedback(NDArray &yd,
    NDArray &y,
    NDArray *baseAction,
    NDArray *meanAction) {
  NDArray::WaitAll();
  NDArray e = yd - y;
  NDArray::WaitAll();
  fb = dot(gains, e);
  NDArray::WaitAll();
  if (baseAction) *baseAction = fb;
  meanAction = NULL;

  NDArray::WaitAll();
  return &fb;
}

int setupFeedback(Controller *controller) {
  log("Setting up LQR feedback");
  std::vector<mx_float> kVec = {
   0.0020, 1.0000, -0.0000, -0.0418, 1.3856, -0.0000, 1.1039, 0.0001, -0.0000, 4.6346, 0.0395, -0.0000,
   -1.0000, 0.0020, -0.0043, -1.3583, -0.0458, -0.0015, 0.0004, 1.1348, -0.0028, -0.0247, 4.6926, -0.0028,
   -0.0035, 0.0000, 0.1773, -0.0070, -0.0001, -0.0188, -0.0000, -0.0010, 1.0131, -0.0002, -0.0142, 0.9842,
   0.0037, -0.0000, -0.9842, 0.0147, 0.0001, -0.9920, 0.0000, 0.0044, 0.2049, 0.0001, 0.2483, 0.1773
  };

  gains = NDArray(kVec, Shape(NUM_INPUTS, NUM_STATES), Context::cpu());
  NDArray::WaitAll();

  controller->feedback = feedback;
  NDArray::WaitAll();

  return SIM_SUCCESS;
}


NDArray *policyFeedback(NDArray &yd,
    NDArray &y,
    NDArray *baseAction,
    NDArray *meanAction) {
  static Shape fbShape(NUM_INPUTS,1);
  NDArray e = yd - y;
  NDArray::WaitAll();
  fb = dot(gains, e);
  NDArray::WaitAll();

  if (baseAction != NULL) *baseAction = fb.Copy(Context::cpu());
  NDArray::WaitAll();

  policy->policyArgs["policyx"] = Concat(yd,y,Shape(1,2*NUM_STATES));
  policy->rebindPolicy();
  NDArray::WaitAll();
#ifdef DEBUG
  t1 = high_resolution_clock::now();
#endif
  policy->policyExec->Forward(false);
  NDArray::WaitAll();
#ifdef DEBUG
  t2 = high_resolution_clock::now();
  policy->executionTimesVec["policySingleForwardPass"].push_back(
      duration_cast<microseconds>(t2-t1).count());
#endif
  if (meanAction != NULL) *meanAction = policy->policyExec->outputs[0]; 
  auto tmp = policy->sample();
  tmp = tmp.Reshape(fbShape);
  //fb = fb + tmp;
  fb = tmp;
  //std::cout << "fb " << fb << std::endl;
  NDArray::WaitAll();
  return &fb;
}

int setupPolicyFeedback(Controller *controller, PolicyFunction *policyFunction) {
  log("Setting up LQR-PPO feedback");
  std::vector<mx_float> kVec = {
   0.0020, 1.0000, -0.0000, -0.0418, 1.3856, -0.0000, 1.1039, 0.0001, -0.0000, 4.6346, 0.0395, -0.0000,
   -1.0000, 0.0020, -0.0043, -1.3583, -0.0458, -0.0015, 0.0004, 1.1348, -0.0028, -0.0247, 4.6926, -0.0028,
   -0.0035, 0.0000, 0.1773, -0.0070, -0.0001, -0.0188, -0.0000, -0.0010, 1.0131, -0.0002, -0.0142, 0.9842,
   0.0037, -0.0000, -0.9842, 0.0147, 0.0001, -0.9920, 0.0000, 0.0044, 0.2049, 0.0001, 0.2483, 0.1773
  };
  gains = NDArray(kVec, Shape(NUM_INPUTS, NUM_STATES), Context::cpu());
  NDArray::WaitAll();

  policy = policyFunction;
  controller->feedback = policyFeedback;
  NDArray::WaitAll();

  return SIM_SUCCESS;
}

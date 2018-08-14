#include <math.h>
//#include <gsl/gsl_blas.h>

#include "dynamics.h"

gsl_matrix *dyn;
gsl_matrix *gains;
std::vector<double> fb;

PolicyFunction policy;

int setupDynamics() {
  double X_u   =  -0.27996;
  double Y_v   =  -0.22566;
  double Z_w   =  -1.2991;
  double L_p   =  -2.5110;
  double M_q   =  -2.4467;
  double N_r   =  -0.4948;
  double X_th  =  -10.067;
  double Y_ph  =   9.8648;
  double L_ph  =  -21.358;
  double M_th  =  -18.664;
  double ph_p  =   0.9655;
  double th_q  =   0.9634;
  double ps_r  =   0.6748;
  double Z_thr =  -39.282;
  double L_la  =   11.468;
  double M_lo  =   9.5711;
  double N_ya  =   3.5647;
  double ph_la =   0.0744;
  double th_lo =   0.0594;
  double ps_ya =   0.0397;

  dyn = gsl_matrix_calloc(NUM_STATES,NUM_STATES+NUM_INPUTS);

  gsl_matrix_set(dyn, 0, 3, 1);
  gsl_matrix_set(dyn, 0, 4, 1);
  gsl_matrix_set(dyn, 1, 3, 1);
  gsl_matrix_set(dyn, 1, 4, 1);
  //gsl_matrix_set(dyn, 0, 3, 1);
  //gsl_matrix_set(dyn, 1, 4, 1);
  gsl_matrix_set(dyn, 2, 5, 1);
  gsl_matrix_set(dyn, 3, 3, X_u);
  gsl_matrix_set(dyn, 3, 10, X_th);
  gsl_matrix_set(dyn, 4, 4, Y_v);
  gsl_matrix_set(dyn, 4, 9, Y_ph);
  gsl_matrix_set(dyn, 5, 5, Z_w);
  gsl_matrix_set(dyn, 5, 15, Z_thr);
  gsl_matrix_set(dyn, 6, 6, L_p);
  gsl_matrix_set(dyn, 6, 9, L_ph);
  gsl_matrix_set(dyn, 6, 12, L_la);
  gsl_matrix_set(dyn, 7, 7, M_q);
  gsl_matrix_set(dyn, 7, 10, M_th);
  gsl_matrix_set(dyn, 7, 13, M_lo);
  gsl_matrix_set(dyn, 8, 8, N_r);
  gsl_matrix_set(dyn, 8, 14, N_ya);
  gsl_matrix_set(dyn, 9, 6, ph_p);
  gsl_matrix_set(dyn, 9, 12, ph_la);
  gsl_matrix_set(dyn, 10, 7, th_q);
  gsl_matrix_set(dyn, 10, 13, th_lo);
  gsl_matrix_set(dyn, 11, 8, ps_r);
  gsl_matrix_set(dyn, 11, 14, ps_ya);

  return SIM_SUCCESS;
}

// Dynamics model from G. Gremillion, S. Humbert paper "System Identification of
// a Quadrotor Micro Air Vehicle"
// Augmented to include position dynamics
//
int dynamics(std::vector<double> *dy, double t, const std::vector<double> *y,
    const std::vector<double> *u) {
  (void) (t); // Avoid unused parameter warning
  gsl_matrix_set(dyn, 0, 3, cos(y->at(11)));
  gsl_matrix_set(dyn, 0, 4, -sin(y->at(11)));
  gsl_matrix_set(dyn, 1, 3, sin(y->at(11)));
  gsl_matrix_set(dyn, 1, 4, cos(y->at(11)));

  std::vector<double> yu = vector_stack(y,u);
  dy->resize(y->size()); 

  //TODO
  cblas_dgemv(CblasRowMajor, CblasNoTrans,
              dyn->size1, dyn->size2,
              1.0, dyn->data, dyn->size2,
              &(yu[0]), 1, 0.0,
              &(dy->at(0)), 1);

  return SIM_SUCCESS;
}

int teardownDynamics() {
  gsl_matrix_free(dyn);
  return SIM_SUCCESS;
}

std::vector<double> *feedback(const std::vector<double> *yd,
    const std::vector<double> *y) {
  std::vector<double> e = vector_sub(yd,y);
  cblas_dgemv(CblasRowMajor, CblasNoTrans,
              gains->size1, gains->size2,
              1.0, gains->data, gains->size2,
              &(e[0]), 1, 0.0,
              &(fb[0]), 1);

  return &fb;
}

int setupFeedback() {
  gains = gsl_matrix_calloc(NUM_INPUTS,NUM_STATES);

  double K[NUM_INPUTS][NUM_STATES] = {
   {0.0020, 1.0000, -0.0000, -0.0418, 1.3856, -0.0000, 1.1039, 0.0001, -0.0000, 4.6346, 0.0395, -0.0000},
   {-1.0000, 0.0020, -0.0043, -1.3583, -0.0458, -0.0015, 0.0004, 1.1348, -0.0028, -0.0247, 4.6926, -0.0028},
   {-0.0035, 0.0000, 0.1773, -0.0070, -0.0001, -0.0188, -0.0000, -0.0010, 1.0131, -0.0002, -0.0142, 0.9842},
   {0.0037, -0.0000, -0.9842, 0.0147, 0.0001, -0.9920, 0.0000, 0.0044, 0.2049, 0.0001, 0.2483, 0.1773}
  };

  for (int i=0; i < NUM_INPUTS; i++) {
    for (int j=0; j < NUM_STATES; j++) {
      gsl_matrix_set(gains,i,j,K[i][j]);
    }
  }

  fb.resize(gains->size1); 

  return SIM_SUCCESS;
}

int teardownFeedback() {
  gsl_matrix_free(gains);
  return SIM_SUCCESS;
}

std::vector<double> *policyFeedback(const std::vector<double> *yd,
    const std::vector<double> *y) {
  std::vector<double> e = vector_sub(yd,y);
  cblas_dgemv(CblasRowMajor, CblasNoTrans,
              gains->size1, gains->size2,
              1.0, gains->data, gains->size2,
              &(e[0]), 1, 0.0,
              &(fb[0]), 1);

  std::vector<double> nnIn = vector_stack(yd,y);
  tiny_dnn::vec_t in = tiny_dnn::convertToVecT(nnIn);

  tiny_dnn::vec_t meanPolicy = policy.policyNet.predict(in);
  policy.probabilityDistribution.set_mean(meanPolicy);
  tiny_dnn::vec_t samplePolicy = policy.probabilityDistribution.sample();

  std::vector<double> policyFb = convertToStdVec(samplePolicy);

	fb = vector_add(&fb,&policyFb);

  return &fb;
}

int setupPolicyFeedback(int dataShape[2]) {
  tiny_dnn::input_layer policyIn(tiny_dnn::shape3d(dataShape[0],1,1));
  tiny_dnn::fully_connected_layer policyHidden1(dataShape[0],100),
                                  policyHidden2(100,10),
                                  policyOut(10,NUM_INPUTS);
  tiny_dnn::tanh_layer policyAct1(100),
                       policyAct2(10);

  // Connect activation layers
  policyHidden1 << policyAct1;
  policyHidden2 << policyAct2;

  // Connect graph
  policyIn << policyHidden1 << policyHidden2 << policyOut;

  tiny_dnn::construct_graph(policy.policyNet, {&policyIn}, {&policyOut});
  policy.oldPolicyNet = policy.policyNet;

  tiny_dnn::input_layer valueIn(tiny_dnn::shape3d(dataShape[0]+dataShape[1],1,1));
  tiny_dnn::fully_connected_layer valueHidden1(dataShape[0]+dataShape[1],100),
                                  valueHidden2(100,10),
                                  valueOut(10,1);
  tiny_dnn::tanh_layer valueAct1(100),
                       valueAct2(10);

  // Connect activation layers
  valueHidden1 << valueAct1;
  valueHidden2 << valueAct2;

  // Connect graph
  valueIn << valueHidden1 << valueHidden2 << valueOut;

  tiny_dnn::construct_graph(policy.valueNet, {&valueIn}, {&valueOut});

  return SIM_SUCCESS;
}

int setupPolicyFeedback(char *policyNetFile, char *valueNetFile) {
  policy.policyNet.load(policyNetFile);
  policy.valueNet.load(valueNetFile);

  return SIM_SUCCESS;
}

int teardownPolicyFeedback() {
  return SIM_SUCCESS;
}

PolicyFunction *getFeedbackPolicy() {
  return &policy;
}


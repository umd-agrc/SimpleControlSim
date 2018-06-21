#pragma once

#include <gsl/gsl_matrix.h>
#include <gsl/gsl_vector.h>

#include <tiny_dnn/tiny_dnn.h>

#include "defines.h"
#include "matrix_vector_ops.h"
#include "genann.h"

#define NUM_STATES 12
#define NUM_INPUTS 4

// Dynamics model from G. Gremillion, S. Humbert paper "System Identification of
// a Quadrotor Micro Air Vehicle" (equation 3 in paper)
// Augmented to include position dynamics in hover
// Dynamics and controls matrix:
// cos(ps)*u  -sin(ps)*v    0     0     0     0     0     0     0     0     0     0     0     0     0     0
// sin(ps)*u  cos(ps)*v     0     0     0     0     0     0     0     0     0     0     0     0     0     0
//  0     0     1     0     0     0     0     0     0     0     0     0     0     0     0     0
//  0     0     0   X_u*u   0     0     0     0     0     0  X_th*th  0     0     0     0     0
//  0     0     0     0   Y_v*v   0     0     0     0  Y_ph*ph  0     0     0     0     0     0
//  0     0     0     0     0   Z_w*w   0     0     0     0     0     0     0     0     0 Z_thr*d_thr
//  0     0     0     0     0     0   L_p*p   0     0  L_ph*ph  0     0 L_la*d_la 0     0     0
//  0     0     0     0     0     0     0   M_q*q   0     0   M_th*th 0     0 M_lo*d_lo 0     0
//  0     0     0     0     0     0     0     0   N_r*r   0     0     0     0     0 N_ya*d_ya 0
//  0     0     0     0     0     0   ph_p*p  0     0     0     0     0 ph_la*d_la 0    0     0
//  0     0     0     0     0     0     0   th_q*q  0     0     0     0     0 th_lo*d_lo 0    0
//  0     0     0     0     0     0     0     0   ps_r*r  0     0     0     0     0 ps_ya*d_ya 0
//
//  Parameters:
//  X_u   =  -0.27996
//  Y_v   =  -0.22566
//  Z_w   =  -1.2991
//  L_p   =  -2.5110
//  M_q   =  -2.4467
//  N_r   =  -0.4948
//  X_th  =  -10.067
//  Y_ph  =   9.8648
//  L_ph  =  -21.358
//  M_th  =  -18.664
//  ph_p  =   0.9655
//  th_q  =   0.9634
//  ps_r  =   0.6748
//  Z_thr =  -39.282
//  L_la  =   11.468
//  M_lo  =   9.5711
//  N_ya  =   3.5647
//  ph_la =   0.0744
//  th_lo =   0.0594
//  ps_ya =   0.0397
// 
// State vector:
//    x y z u v w p q r phi theta psi
//
// Control vector:
//    del_lat del_lon del_yaw del_thrust
int dynamics(gsl_vector *dy, double t, const gsl_vector *y, const gsl_vector *u);
int setupDynamics();
int teardownDynamics();

gsl_vector *feedback(gsl_vector *yd, gsl_vector *y);
int setupFeedback();
int teardownFeedback();

gsl_vector *nnFeedback(gsl_vector *yd, gsl_vector *y);
int setupNnFeedback(int dataShape[2]);
int setupNnFeedback(FILE *in);
int teardownNnFeedback();
genann *getFeedbackNn();

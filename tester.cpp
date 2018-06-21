#include <gsl/gsl_errno.h>
#include <gsl/gsl_matrix.h>
#include <gsl/gsl_vector.h>
#include <gsl/gsl_odeiv2.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <math.h>
#include <pthread.h>
#include <semaphore.h>
#include <sys/types.h>
#include <deque>

#include "tester.h"
#include "genann.h"
#include "testDynamics.h"
#include "diff.h"
#include "data.h"

int main(int argc, char *argv[]) {
  int numSteps = 300;
  double t[numSteps];
  double stepSize;
  t[0] = 0;

  // Initialize vehicle and controller
  VehicleState *vehicle = new VehicleState;
  Controller *controller = new Controller;
  setupSystem(vehicle,controller);

  gsl_vector *y_next = gsl_vector_calloc(NUM_STATES);
  gsl_vector *rk_e_next = gsl_vector_calloc(NUM_STATES);

  FILE *dataFile = fopen("nn_control.csv","w");

  logHeader(dataFile,"some stuff to test header action\n");
  logTime(dataFile,t[0],",","a");
  logVector(dataFile,vehicle->y,",","a",true);
  logVector(dataFile,rk_e_next,",","n",false);
  for (int i=1; i < numSteps; i++) {
    rungeKutteAdaptiveStep(&testDynamics,
                           t[i-1],
                           y_next,
                           rk_e_next,
                           vehicle,
                           controller,
                           &stepSize,
                           -1);

    t[i] = t[i-1] + stepSize;

    logTime(dataFile,t[i],",","a");
    logVector(dataFile,y_next,",","a",true);
    logVector(dataFile,rk_e_next,",","n",false);

    gsl_vector_memcpy(vehicle->y,y_next);
  }

  fclose(dataFile);

  teardownSystem(vehicle,controller);
  delete vehicle;
  delete controller;
  return 0;
}

int setupSystem(VehicleState *vehicle, Controller *controller) {
  vehicle->yd = gsl_vector_calloc(NUM_STATES);
  vehicle->y = gsl_vector_calloc(NUM_STATES);
  controller->feedback = &testFeedback;

  gsl_vector_set(vehicle->y,0,100);

  setupTestDynamics();
  setupTestFeedback();

  return SIM_SUCCESS;
}

int teardownSystem(VehicleState *vehicle, Controller *controller) {
  gsl_vector_free(vehicle->yd);
  gsl_vector_free(vehicle->y);

  teardownTestDynamics();
  teardownTestFeedback();
  
  return SIM_SUCCESS;
}

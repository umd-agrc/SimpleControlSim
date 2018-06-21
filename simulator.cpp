#include "simulator.h"

const char *training_data = "./data/lqrData.txt";
//double *input, *output;
std::vector<tiny_dnn::vec_t> input, output;
int samples = 0;

void *simulate(void *var) {
  std::deque<char*> *sendQueue = static_cast<std::deque<char*>*>(var);
  char sendBuff[1025];
  bool shouldExit = false;
  int numSteps = 4200;
  double t[numSteps];
  double stepSize;
  t[0] = 0;

  int dataShape[2];

  /* Load the training data and saved nn from file. */
  load_data(training_data,dataShape,&input,&output,&samples);

  // Initialize vehicle and controller
  VehicleState *vehicle = new VehicleState;
  Controller *controller = new Controller;
  //setupSystem(vehicle,controller,dataShape); // to get new nn
  // Load nn from file
  //FILE *feedbackNnPrev = fopen("./nn/feedbackNn20Loops.txt","r");
  setupSystem(vehicle,controller,feedbackNnPrev);
  //fclose(feedbackNnPrev);
  //genann *feedbackNn = getFeedbackNn();

  /* Train the network with backpropagation. */
  /*
  FILE *feedbackNnFile = fopen("./nn/feedbackNn.txt","w");
  int loops = 5;
  printf("Training for %d loops over data.\n", loops);
  for (int i = 0; i < loops; ++i) {
    for (int j = 0; j < samples; ++j) {
      genann_train(feedbackNn, input + j*dataShape[0], output + j*dataShape[1], .01);
    }
  }
  genann_write(feedbackNn,feedbackNnFile);
  */

  gsl_vector *y_next = gsl_vector_calloc(NUM_STATES);
  gsl_vector *rk_e_next = gsl_vector_calloc(NUM_STATES);

  FILE *dataFile = fopen("data/lqrNn20Loops.csv","w");

  logHeader(dataFile,"t,xd,yd,zd,ud,vd,wd,pd,qd,rd,phid,thetad,psid,x,y,z,u,v,w,p,q,r,phi,theta,psi,del_lat,del_lon,del_yaw,del_thrust\n");
  logTime(dataFile,t[0],",","a");
  logVector(dataFile,vehicle->yd,",","a",true);
  logVector(dataFile,vehicle->y,",","a",true);
  logVector(dataFile,controller->feedback(vehicle->yd,vehicle->y),",","n",false);
  int i=1;
  while (!shouldExit && i < numSteps) {
    rungeKutteAdaptiveStep(&dynamics,
                           t[i-1],
                           y_next,
                           rk_e_next,
                           vehicle,
                           controller,
                           &stepSize,
                           -1);

    t[i] = t[i-1] + stepSize;

    gsl_vector_memcpy(vehicle->y,y_next);

    logTime(dataFile,t[i],",","a");
    logVector(dataFile,vehicle->yd,",","a",true);
    logVector(dataFile,vehicle->y,",","a",true);
    logVector(dataFile,controller->feedback(vehicle->yd,vehicle->y),",","n",false);

    if (formNextMsg(sendBuff) == SIM_SUCCESS)
      sendQueue->push_back(sendBuff);
    //usleep(15000);

    i++;
  }

  printf("done!!\n");
  fclose(dataFile);
  //fclose(feedbackNnFile);

  strcpy(sendBuff,SHUTDOWN_MESSAGE);
  sendQueue->push_back(sendBuff);

  teardownSystem(vehicle,controller);
  delete vehicle;
  delete controller;

  return (void *)0;
}

int formNextMsg(char *msg) {
  static int i=0;
  sprintf(msg,"%d\n",++i); 
  return SIM_SUCCESS;
}

int setupSystem(VehicleState *vehicle, Controller *controller, int dataShape[2]) {
  vehicle->yd = gsl_vector_calloc(NUM_STATES);
  vehicle->y = gsl_vector_calloc(NUM_STATES);

  controller->feedback = &nnFeedback;

  // Set initial and desired state vector
  //    x y z u v w p q r phi theta psi
  gsl_vector_set(vehicle->yd, 0, 0);
  gsl_vector_set(vehicle->yd, 1, 0);
  gsl_vector_set(vehicle->yd, 2, 0);
  gsl_vector_set(vehicle->yd, 3, 0);
  gsl_vector_set(vehicle->yd, 4, 0);
  gsl_vector_set(vehicle->yd, 5, 0);
  gsl_vector_set(vehicle->yd, 6, 0);
  gsl_vector_set(vehicle->yd, 7, 0);
  gsl_vector_set(vehicle->yd, 8, 0);
  gsl_vector_set(vehicle->yd, 9, 0);
  gsl_vector_set(vehicle->yd, 10, 0);
  gsl_vector_set(vehicle->yd, 11, 0);

  gsl_vector_set(vehicle->y, 0, 1);
  gsl_vector_set(vehicle->y, 1, 1);
  gsl_vector_set(vehicle->y, 2, 1);
  gsl_vector_set(vehicle->y, 3, 1);
  gsl_vector_set(vehicle->y, 4, 1);
  gsl_vector_set(vehicle->y, 5, 1);
  gsl_vector_set(vehicle->y, 6, 1);
  gsl_vector_set(vehicle->y, 7, 1);
  gsl_vector_set(vehicle->y, 8, 1);
  gsl_vector_set(vehicle->y, 9, 1);
  gsl_vector_set(vehicle->y, 10, 1);
  gsl_vector_set(vehicle->y, 11, 1);

  setupDynamics();
  setupNnFeedback(dataShape);
  
  return SIM_SUCCESS;
}

int setupSystem(VehicleState *vehicle, Controller *controller, FILE *in) {
  vehicle->yd = gsl_vector_calloc(NUM_STATES);
  vehicle->y = gsl_vector_calloc(NUM_STATES);

  controller->feedback = &nnFeedback;

  // Set initial and desired state vector
  //    x y z u v w p q r phi theta psi
  gsl_vector_set(vehicle->yd, 0, 0);
  gsl_vector_set(vehicle->yd, 1, 0);
  gsl_vector_set(vehicle->yd, 2, 0);
  gsl_vector_set(vehicle->yd, 3, 0);
  gsl_vector_set(vehicle->yd, 4, 0);
  gsl_vector_set(vehicle->yd, 5, 0);
  gsl_vector_set(vehicle->yd, 6, 0);
  gsl_vector_set(vehicle->yd, 7, 0);
  gsl_vector_set(vehicle->yd, 8, 0);
  gsl_vector_set(vehicle->yd, 9, 0);
  gsl_vector_set(vehicle->yd, 10, 0);
  gsl_vector_set(vehicle->yd, 11, 0);

  gsl_vector_set(vehicle->y, 0, 1);
  gsl_vector_set(vehicle->y, 1, 1);
  gsl_vector_set(vehicle->y, 2, 1);
  gsl_vector_set(vehicle->y, 3, 1);
  gsl_vector_set(vehicle->y, 4, 1);
  gsl_vector_set(vehicle->y, 5, 1);
  gsl_vector_set(vehicle->y, 6, 1);
  gsl_vector_set(vehicle->y, 7, 1);
  gsl_vector_set(vehicle->y, 8, 1);
  gsl_vector_set(vehicle->y, 9, 1);
  gsl_vector_set(vehicle->y, 10, 1);
  gsl_vector_set(vehicle->y, 11, 1);

  setupDynamics();
  setupNnFeedback(in);
  
  return SIM_SUCCESS;
}

int teardownSystem(VehicleState *vehicle, Controller *controller) {
  gsl_vector_free(vehicle->yd);
  gsl_vector_free(vehicle->y);

  teardownDynamics();
  teardownNnFeedback();

  //teardownData(input,output);
  
  return SIM_SUCCESS;
}

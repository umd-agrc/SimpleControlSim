#include "simulator.h"

void *simulate(void *var) {
  std::deque<char*> *sendQueue = static_cast<std::deque<char*>*>(var);
  bool shouldExit = false;
  char sendBuff[1025];
  int dataShape[2];

  // Initialize vehicle and controller
  VehicleState *vehicle = new VehicleState;
  Controller *controller = new Controller;
  //int sysType = NN_TYPE;
  //int sysType = LQR_TYPE;
  //TODO systype as command line argument
  int sysType = POLICY_TYPE;
  if (sysType == LQR_TYPE) {
    setupSystem(vehicle,controller,dataShape,sysType);

    char dataFilename[1024];
    strcpy(dataFilename,"./data/lqrCtl");
    strcat(dataFilename,".csv");
    testFeedbackControl(&shouldExit,dataFilename,sendQueue,vehicle,controller,1000);

    strcpy(sendBuff,SHUTDOWN_MESSAGE);
    sendQueue->push_back(sendBuff);
  } else if (sysType == POLICY_TYPE) {	
    // Load training data if any is cached
    char *trainingDataFile = "data/policy/train/policyTrainingData.txt";
    if (fileExists(trainingDataFile)) {
      //load_data(dqnTrainingDataFile,dataShape,&input,&output,&samples);
    }

    char *policyNetFile = "nn/policy/feedbackPolicyNet";
    char *valueNetFile = "nn/policy/feedbackValueNet";
    if (fileExists(policyNetFile) && fileExists(valueNetFile))
      setupSystem(vehicle,controller,policyNetFile,valueNetFile,sysType);
    else {
      dataShape[0] = NUM_STATES*2; dataShape[1] = NUM_INPUTS;
      setupSystem(vehicle,controller,dataShape,sysType);
    }
    
		srand(time(NULL));
    PolicyFunction *policy = getFeedbackPolicy();
		testPolicyFeedbackControl(&shouldExit,policyNetFile,valueNetFile,trainingDataFile,
      sendQueue,policy,vehicle,controller);
    strcpy(sendBuff,SHUTDOWN_MESSAGE);
    sendQueue->push_back(sendBuff);
	}

  printf("test done!!\n");

  teardownSystem(vehicle,controller,sysType);
  delete vehicle;
  delete controller;

  return (void *)0;
}

int resetSystem(VehicleState *vehicle, Controller *controller) {
  // Set initial and desired state vector
  //    x y z u v w p q r phi theta psi
  for (auto it = vehicle->yd.begin(); it != vehicle->yd.end(); it++){
    *it = 0; 
  } 
  for (auto it = vehicle->y.begin(); it != vehicle->y.end(); it++){
    *it = 1; 
  } 
  return SIM_SUCCESS;
}

int setupSystem(VehicleState *vehicle, Controller *controller, int dataShape[2], int type) {
  printf("Setting up system\n");

  // Set initial and desired state vector
  //    x y z u v w p q r phi theta psi
  vehicle->yd.resize(NUM_STATES);
  vehicle->y.resize(NUM_STATES);
  resetSystem(vehicle,controller);

  setupDynamics();

  if (type == LQR_TYPE) {
    controller->feedback = feedback;
    setupFeedback();
  } else if (type == POLICY_TYPE) {
    controller->feedback = policyFeedback;
    setupPolicyFeedback(dataShape);
  }
  
  return SIM_SUCCESS;
}

int setupSystem(VehicleState *vehicle, Controller *controller,
    char *policyNetFile, char *valueNetFile, int type) {
  printf("Setting up system\n");
  // Set initial and desired state vector
  //    x y z u v w p q r phi theta psi
  vehicle->yd.resize(NUM_STATES);
  vehicle->y.resize(NUM_STATES);
  resetSystem(vehicle,controller);

  setupDynamics();

  if (type == LQR_TYPE) {
    //FIXME return error or setup input from file
    controller->feedback = feedback;
    setupFeedback();
  } else if (type == POLICY_TYPE) {
    controller->feedback = policyFeedback;
    setupPolicyFeedback(policyNetFile, valueNetFile);
  }
  
  return SIM_SUCCESS;
}

int teardownSystem(VehicleState *vehicle, Controller *controller, int type) {
  teardownDynamics();

  if (type == LQR_TYPE) {
    teardownFeedback();
  } else if (type == POLICY_TYPE) {
    teardownPolicyFeedback();
	}

  return SIM_SUCCESS;
}


void testPolicyFeedbackControl(bool *shouldExit, char *policyNetFile,
    char *valueNetFile, char *trainingDataFile,
		std::deque<char*> *sendQueue, PolicyFunction *policy,
    VehicleState *vehicle, Controller *controller) {
  //char sendBuff[1025];
	int totalEpochs = 5;

  int epoch = 0;
  while (!*shouldExit && epoch < totalEpochs) {
    // Test controller
    testFeedbackControl(shouldExit, NULL, sendQueue,
      vehicle, controller, 1000);

    // Update controller policy
    policyUpdate(shouldExit, policyNetFile, valueNetFile,
      trainingDataFile, sendQueue, policy, vehicle, controller);

		epoch++;
	}
}



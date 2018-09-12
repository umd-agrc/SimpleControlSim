#include "simulator.h"

using namespace mxnet::cpp;

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
    setupSystem(vehicle,controller);

    char dataFilename[1024];
    strcpy(dataFilename,"./data/lqrCtl");
    strcat(dataFilename,".csv");
    //TODO fix this
    //testFeedbackControl(&shouldExit,dataFilename,sendQueue,vehicle,controller,1000);

    strcpy(sendBuff,SHUTDOWN_MESSAGE);
    sendQueue->push_back(sendBuff);
  } else if (sysType == POLICY_TYPE) {	
    log("Creating policy gradient simulation");

    PolicyFunction policy;
    //TODO load files
    if (0) {
      log("Loading value and policy nets from files");
      //setupSystem(vehicle,controller,policyNetFile,valueNetFile,sysType);
    } else {
      log("Setting up value and policy nets from scratch");
      setupSystem(vehicle,controller,&policy);
    }
    
		srand(time(NULL));
		testPolicyFeedbackControl(&shouldExit,sendQueue,policy,vehicle,controller);
    /*
    strcpy(sendBuff,SHUTDOWN_MESSAGE);
    sendQueue->push_back(sendBuff);
    */
    MXNotifyShutdown();
	}

  log("Test done!!");

  delete vehicle;
  delete controller;

  return (void *)0;
}

int resetSystem(VehicleState *vehicle, Controller *controller) {
  // Set initial and desired state vector
  //    x y z u v w p q r phi theta psi
  vehicle->yd = zeros(Shape(NUM_STATES,1));
  vehicle->y = ones(Shape(NUM_STATES,1));
  return SIM_SUCCESS;
}

int setupSystem(VehicleState *vehicle, Controller *controller) {
  log("Setting up system for LQR control");
  // Set initial and desired state vector
  //    x y z u v w p q r phi theta psi
  resetSystem(vehicle,controller);
  setupDynamics();
  controller->feedback = feedback;
  setupFeedback();
 
  return SIM_SUCCESS;
}

int setupSystem(VehicleState *vehicle, Controller *controller, PolicyFunction *policy) {
  log("Setting up system for LQR-PPO control");
  // Set initial and desired state vector
  //    x y z u v w p q r phi theta psi
  resetSystem(vehicle,controller);
  setupDynamics();
  setupPolicyFeedback(controller, policy);

  return SIM_SUCCESS;
}

void testPolicyFeedbackControl(bool *shouldExit,
		std::deque<char*> *sendQueue, PolicyFunction &policy,
    VehicleState *vehicle, Controller *controller) {
	int numEpoch = 9;

  setupLoss(policy);

  int epoch = 0;
  int numSteps = 10;

  //TODO save network configuration after each epoch
  while (!*shouldExit && epoch < numEpoch) {
    log("Performing LQR-PPO test epoch ", epoch, ""); 
    // Update controller policy
    policyUpdate(shouldExit, sendQueue, policy, vehicle, controller, numSteps);
    if (epoch % 3 == 0) {
      log("Logging trajectory of epoch ", epoch); fflush(stdout);
      logNDArrayMap("data/policy/arr/",
                    "-epoch" + std::to_string(epoch) + ".ndarray",
                    policy.trajSegment);
    }

		epoch++;
	}

  policy.saveSym("data/policy/","-symbol.json");
#ifdef DEBUG
  for (auto vec : policy.executionTimesVec) {
    policy.executionTimes[vec.first] =
      NDArray(vec.second,Shape(vec.second.size(),1),Context::cpu());
  }
  //NDArray::Save("data/policy/" + policy.executionTimesString,
  //              policy.executionTimes);

  logNDArrayMap("data/policy/dbg/execution_times/",
                ".ndarray",
                policy.executionTimes);
#endif

  log("Finished LQR-PPO control test");

  NDArray::WaitAll();
}



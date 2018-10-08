#include "simulator.h"

using namespace mxnet::cpp;

void *simulate(void *var) {
  std::deque<char*> *sendQueue = static_cast<std::deque<char*>*>(var);
  bool shouldExit = false;
  char sendBuff[1025];
  int dataShape[2];

  setupRandomDistribution();

  // Initialize vehicle and controller
  VehicleState *vehicle = new VehicleState;
  Controller *controller = new Controller;
  //TODO systype as command line argument
  //int sysType = LQR_TYPE;
  int sysType = POLICY_TYPE;
  //int sysType = TRAIN_TYPE;
  if (sysType == LQR_TYPE) {
    setupSystem(vehicle,controller);

    log("Testing LQR feedback");
    std::map<std::string,NDArray> trajectory;
    NDArray::WaitAll();
    testFeedbackControl(&shouldExit,NULL,NULL,&trajectory,sendQueue,vehicle,controller,500);
    logNDArrayMap("data/lqr/arr/",
                  ".ndarray",
                  trajectory);

    //strcpy(sendBuff,SHUTDOWN_MESSAGE);
    //sendQueue->push_back(sendBuff);
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
    
		testPolicyFeedbackControl(&shouldExit,sendQueue,policy,vehicle,controller);
    /*
    strcpy(sendBuff,SHUTDOWN_MESSAGE);
    sendQueue->push_back(sendBuff);
    */
    MXNotifyShutdown();
	} else if (sysType == TRAIN_TYPE) {	
    log("Training policy gradient");

    PolicyFunction policy;
		trainPolicyFeedbackControl(&shouldExit,sendQueue,policy,vehicle,controller);

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
  NDArray::WaitAll();
  return SIM_SUCCESS;
}

int setupSystem(VehicleState *vehicle, Controller *controller) {
  log("Setting up system for LQR control");
  // Set initial and desired state vector
  //    x y z u v w p q r phi theta psi
  resetSystem(vehicle,controller);
  NDArray::WaitAll();
  setupDynamics();
  setupFeedback(controller);
 
  return SIM_SUCCESS;
}

int setupSystem(VehicleState *vehicle, Controller *controller, PolicyFunction *policy) {
  log("Setting up system for LQR-PPO control");
  // Set initial and desired state vector
  //    x y z u v w p q r phi theta psi
  resetSystem(vehicle,controller);
  NDArray::WaitAll();
  setupDynamics();
  setupPolicyFeedback(controller, policy);

  return SIM_SUCCESS;
}

void testPolicyFeedbackControl(bool *shouldExit,
		std::deque<char*> *sendQueue, PolicyFunction &policy,
    VehicleState *vehicle, Controller *controller) {
	int numEpoch = 50;
  setupLoss(policy);

  int epoch = 0;
  int numSteps = 100;

  //TODO save network configuration after each epoch
  while (!*shouldExit && epoch < numEpoch) {
    resetSystem(vehicle,controller);
    log("Performing LQR-PPO test epoch ", epoch, ""); 
    // Update controller policy
    policyUpdate(shouldExit, sendQueue, policy, vehicle, controller, numSteps);
    if (epoch % 2 == 0) {
      log("Logging trajectory of epoch ", epoch); fflush(stdout);
      logNDArrayMap("data/policy/arr/",
                    "-epoch" + std::to_string(epoch) + ".ndarray",
                    policy.trajSegment);
      NDArray::Save(
          "data/policy/train/trajSegment-epoch" + std::to_string(epoch) + ".ndbin",
          policy.trajSegment);

      NDArray::Save(
          "data/policy/train/policyNet-epoch" + std::to_string(epoch) + ".ndbin",
          policy.policyExec->arg_dict());

      NDArray::Save(
          "data/policy/train/valueNet-epoch" + std::to_string(epoch) + ".ndbin",
          policy.valueExec->arg_dict());
    }
		epoch++;
	}

  policy.saveSym("data/policy/","-symbol.json");
#ifdef DEBUG
  for (auto vec : policy.executionTimesVec) {
    policy.executionTimes[vec.first] =
      NDArray(vec.second,Shape(vec.second.size(),1),Context::cpu());
  }

  logNDArrayMap("data/policy/dbg/execution_times/",
                ".ndarray",
                policy.executionTimes);
#endif

  NDArray::Save(
      "data/policy/train/trajSegment-recent.ndbin",
      policy.trajSegment);

  NDArray::Save(
      "data/policy/train/policyNet-recent.ndbin",
      policy.policyExec->arg_dict());

  NDArray::Save(
      "data/policy/train/valueNet-recent.ndbin",
      policy.valueExec->arg_dict());

  log("Finished LQR-PPO control test");

  NDArray::WaitAll();
}


void trainPolicyFeedbackControl(bool *shouldExit,
		std::deque<char*> *sendQueue, PolicyFunction &policy,
    VehicleState *vehicle, Controller *controller) {
	int numEpoch = 10;

  policy.trajSegment = NDArray::LoadToMap("data/policy/train/trajSegment-recent.ndbin");

  auto policyArgs = NDArray::LoadToMap("data/policy/train/policyNet-recent.ndbin");
  auto valueArgs = NDArray::LoadToMap("data/policy/train/valueNet-recent.ndbin");

  for (auto a : policyArgs) {
    std::cout << a.first << "\n" << a.second << std::endl;
  }
  policy.policyExec = policy.policyNet.SimpleBind(Context::cpu(),policyArgs); 
  policy.valueExec = policy.valueNet.SimpleBind(Context::cpu(),valueArgs); 

  setupLoss(policy);

  int epoch = 0;

  while (!*shouldExit && epoch < numEpoch) {
    log("Performing LQR-PPO train epoch ", epoch, ""); 
    epoch++;
  }

  NDArray::WaitAll();
}

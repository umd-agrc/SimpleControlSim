#include "runner.h"

using namespace mxnet::cpp;

void testFeedbackControl(bool *shouldExit,
    char *dataFilename, PolicyFunction &policy,
    std::deque<char*> *sendQueue, VehicleState *vehicle, Controller *controller,
    int numSteps) {
	NDArray observations = NDArray(Shape(numSteps,2*NUM_STATES),Context::cpu(),false),
          actions = NDArray(Shape(numSteps,NUM_INPUTS),Context::cpu(),false),
          observationsAndActions =
            NDArray(Shape(numSteps,2*NUM_STATES+NUM_INPUTS),Context::cpu(),false),
          baseActions = NDArray(Shape(numSteps,NUM_INPUTS),Context::cpu(),false),
          nextBaseAction,
          meanActions = NDArray(Shape(numSteps,NUM_INPUTS),Context::cpu(),false),
          nextMeanAction;
  NDArray y_next, rk_e_next;
  NDArray *u;
  char sendBuff[1025];
  mx_float stepSize;
  mx_float t[numSteps];
  t[0] = 0;
	FILE *dataFile = NULL;
  size_t currObservationsSz = 2*NUM_STATES,
         currObservationsAndActionsSz = 2*NUM_STATES+NUM_INPUTS;
  log("Testing feedback control");

  u = controller->feedback(vehicle->yd,vehicle->y,&nextBaseAction,&nextMeanAction);
  auto currObservations = Concat(vehicle->yd,vehicle->y,Shape(1,currObservationsSz));
  observations.SetData(0,0,currObservations);
  actions.SetData(0,0,*u);
  auto currObservationsAndActions =
    Concat(currObservations,*u,Shape(1,currObservationsAndActionsSz));
  baseActions.SetData(0,0,nextBaseAction);
  meanActions.SetData(0,0,nextMeanAction);

  int i=1;
  bool reset = true;
  while (!*shouldExit && i < numSteps) {
    rungeKutteAdaptiveStep(&dynamics,
                           t[i-1],
                           &y_next,
                           &rk_e_next,
                           vehicle,
                           controller,
                           &stepSize,
                           -1,
                           reset);
    reset = false;

    t[i] = t[i-1] + stepSize;

    vehicle->y = y_next;
    u = controller->feedback(vehicle->yd,vehicle->y,&nextBaseAction,&nextMeanAction);
    currObservations = Concat(vehicle->yd,vehicle->y,Shape(1,currObservationsSz));
    observations.SetData(i,0,currObservations);
    actions.SetData(i,0,*u);
    currObservationsAndActions =
      Concat(currObservations,*u,Shape(1,currObservationsAndActionsSz));
    baseActions.SetData(i,0,nextBaseAction);
    meanActions.SetData(i,0,nextMeanAction);

    /*
    if (formNextMsg(sendBuff) == SIM_SUCCESS)
      sendQueue->push_back(sendBuff);
    //usleep(15000);
    */

    i++;
  }

  // Set array map for policy update
  policy.baseActions = baseActions;
  policy.trajSegment["observation"] = observations;
  policy.trajSegment["action"] = actions - baseActions;
  policy.trajSegment["observationAndAction"] = observationsAndActions;
  policy.trajSegment["meanCtl"] = meanActions;
  policy.trajSegment["std"] = policy.getStd(Shape(numSteps,NUM_INPUTS));

  policy.oldPolicyArgs["policyx"] = observations;
  policy.rebindOldPolicy();
  policy.oldPolicyExec->Forward(false);
  policy.trajSegment["oldMeanCtl"] = policy.oldPolicyExec->outputs[0];
  policy.trajSegment["oldStd"] = policy.getStd(Shape(numSteps,NUM_INPUTS));
  policy.trajSegment["oldAction"] = policy.trajSegment["oldMeanCtl"]
    + policy.trajSegment["oldStd"]*policy.getRand(Shape(numSteps,NUM_INPUTS));

	if (dataFilename) fclose(dataFile);
}

int formNextMsg(char *msg) {
  static int i=0;
  sprintf(msg,"%d\n",++i); 
  return SIM_SUCCESS;
}

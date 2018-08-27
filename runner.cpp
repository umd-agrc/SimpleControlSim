#include "runner.h"

using namespace mxnet::cpp;

void testFeedbackControl(bool *shouldExit,
    char *dataFilename, PolicyFunction &policy,
    std::deque<char*> *sendQueue, VehicleState *vehicle, Controller *controller,
    int numSteps) {
	std::vector<mx_float> observations,
                        actions,
	                      observationsAndActions,
                        baseActions,
                        nextBaseAction,
                        meanActions,
                        nextMeanAction;
  std::vector<mx_float> y_next, rk_e_next;
  std::vector<mx_float> *u;
  char sendBuff[1025];
  mx_float stepSize;
  //double t[4200]; int numSteps = 4200;
  mx_float t[numSteps];
  t[0] = 0;
	FILE *dataFile = NULL;

	if (dataFilename) {
		dataFile = fopen(dataFilename,"w");

		logHeader(dataFile,"t,xd,yd,zd,ud,vd,wd,pd,qd,rd,phid,thetad,psid,x,y,z,u,v,w,p,q,r,phi,theta,psi,del_lat,del_lon,del_yaw,del_thrust\n");
		logTime(dataFile,t[0],",","a");
		logVector(dataFile,&vehicle->yd,",","a",true);
		logVector(dataFile,&vehicle->y,",","a",true);
		logVector(dataFile,controller->feedback(&vehicle->yd,&vehicle->y,NULL,NULL),",","n",false);
	}

  u = controller->feedback(&vehicle->yd,&vehicle->y,&nextBaseAction,&nextMeanAction);
  /*
  auto currObservations = vector_stack(&vehicle->yd,&vehicle->y);
  observations.insert(
      observations.end(),currObservations.begin(),currObservations.end());
  actions.insert(actions.end(),u->begin(),u->end());
  auto tmp = vector_stack(&currObservations,u);
  observationsAndActions.insert(observationsAndActions.end(),tmp.begin(),tmp.end());
  baseActions.insert(baseActions.end(),nextBaseAction.begin(),nextBaseAction.end());
  meanActions.insert(meanActions.end(),nextMeanAction.begin(),nextMeanAction.end());

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
    u = controller->feedback(&vehicle->yd,&vehicle->y,&nextBaseAction,&nextMeanAction);
    auto currObservations = vector_stack(&vehicle->yd,&vehicle->y);
    observations.insert(
        observations.end(),currObservations.begin(),currObservations.end());
    actions.insert(actions.end(),u->begin(),u->end());
    auto tmp = vector_stack(&currObservations,u);
    observationsAndActions.insert(observationsAndActions.end(),tmp.begin(),tmp.end());
    baseActions.insert(baseActions.end(),nextBaseAction.begin(),nextBaseAction.end());
    meanActions.insert(meanActions.end(),nextMeanAction.begin(),nextMeanAction.end());

		if (dataFilename) {
			logTime(dataFile,t[i],",","a");
			logVector(dataFile,&vehicle->yd,",","a",true);
			logVector(dataFile,&vehicle->y,",","a",true);
			logVector(dataFile,u,",","n",false);
		}

    if (formNextMsg(sendBuff) == SIM_SUCCESS)
      sendQueue->push_back(sendBuff);
    //usleep(15000);

    i++;
  }
  */

  /*
  // Set array map for policy update
  policy.trajSegment["observation"] =
    NDArray(observations,Shape(numSteps,2*NUM_STATES),Context::cpu());
  policy.trajSegment["action"] =
    NDArray(actions,Shape(numSteps,NUM_INPUTS),Context::cpu());
  policy.trajSegment["observationAndAction"] =
    NDArray(observationsAndActions,Shape(numSteps,2*NUM_STATES + NUM_INPUTS),Context::cpu());
  policy.trajSegment["mean"] =
    NDArray(meanActions,Shape(numSteps,NUM_INPUTS),Context::cpu());
  policy.baseActions =
    NDArray(baseActions,Shape(numSteps,NUM_INPUTS),Context::cpu());
  policy.trajSegment["std"] = policy.getStd(Shape(numSteps,NUM_INPUTS));
  policy.trajSegment["logstd"] = log(policy.trajSegment["std"]);

  policy.policyArgs["observation"].CopyTo(&policy.oldPolicyArgs["observation"]);
  policy.oldPolicyExec->Forward(false);
  policy.oldTrajSegment["mean"] =
    policy.oldPolicyExec->outputs[0] + policy.baseActions;
  policy.oldTrajSegment["std"] = policy.getStd(Shape(numSteps,NUM_INPUTS));
  policy.oldTrajSegment["logstd"] = log(policy.oldTrajSegment["std"]);
  policy.oldTrajSegment["action"] =
    policy.oldTrajSegment["mean"]
    + policy.oldTrajSegment["std"]*policy.getRand(Shape(numSteps,NUM_INPUTS));

	if (dataFilename) fclose(dataFile);
  */
}

int formNextMsg(char *msg) {
  static int i=0;
  sprintf(msg,"%d\n",++i); 
  return SIM_SUCCESS;
}

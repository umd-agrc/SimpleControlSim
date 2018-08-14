#include "runner.h"

std::vector<DataPoint> testFeedbackControl(bool *shouldExit,
    char *dataFilename, std::deque<char*> *sendQueue,
    VehicleState *vehicle, Controller *controller,
    int numSteps) {
	std::vector<DataPoint> trajectory;
  std::vector<double> y_next, rk_e_next;
  std::vector<double> *u;
  char sendBuff[1025];
  double stepSize;
  //double t[4200]; int numSteps = 4200;
  double t[numSteps];
  t[0] = 0;
	FILE *dataFile = NULL;

	if (dataFilename) {
		dataFile = fopen(dataFilename,"w");

		logHeader(dataFile,"t,xd,yd,zd,ud,vd,wd,pd,qd,rd,phid,thetad,psid,x,y,z,u,v,w,p,q,r,phi,theta,psi,del_lat,del_lon,del_yaw,del_thrust\n");
		logTime(dataFile,t[0],",","a");
		logVector(dataFile,&vehicle->yd,",","a",true);
		logVector(dataFile,&vehicle->y,",","a",true);
		logVector(dataFile,controller->feedback(&vehicle->yd,&vehicle->y),",","n",false);
	}

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
    u = controller->feedback(&vehicle->yd,&vehicle->y);

		trajectory.push_back(DataPoint(vector_stack(&vehicle->yd,&vehicle->y),*u));

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
  
	if (dataFilename) fclose(dataFile);

	return trajectory;
}

int formNextMsg(char *msg) {
  static int i=0;
  sprintf(msg,"%d\n",++i); 
  return SIM_SUCCESS;
}



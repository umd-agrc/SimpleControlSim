#pragma once

#include <stdlib.h>
#include <unistd.h>
#include <deque>
#include <time.h>
#include <math.h>

#include <tiny_dnn/tiny_dnn.h>

#include "defines.h"
#include "data.h"
#include "dynamics.h"
#include "diff.h"
#include "runner.h"
#include "policy.h"

#define DQN_TRAINING_POINTS 10
#define DQN_POINTS_TO_ADD 1

void *simulate(void *var);

int resetSystem(VehicleState *vehicle, Controller *controller);
int setupSystem(VehicleState *vehicle, Controller *controller, int dataShape[2], int type);
int setupSystem(VehicleState *vehicle, Controller *controller,
    char *policyNetFile, char *valueNetFile, int type);
int teardownSystem(VehicleState *vehicle, Controller *controller, int type);

void testPolicyFeedbackControl(bool *shouldExit, char *policyNetFile,
    char *valueNetFile, char *trainingDataFile,
		std::deque<char*> *sendQueue, PolicyFunction *policy,
    VehicleState *vehicle, Controller *controller);

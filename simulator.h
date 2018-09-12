#pragma once

#include <stdlib.h>
#include <unistd.h>
#include <deque>
#include <time.h>
#include <math.h>

#include "defines.h"
#include "data.h"
#include "dynamics.h"
#include "diff.h"
#include "runner.h"
#include "policy_eval.h"

#define DQN_TRAINING_POINTS 10
#define DQN_POINTS_TO_ADD 1

void *simulate(void *var);

int resetSystem(VehicleState *vehicle, Controller *controller);
int setupSystem(VehicleState *vehicle, Controller *controller);
int setupSystem(VehicleState *vehicle, Controller *controller, PolicyFunction *policy);

void testPolicyFeedbackControl(bool *shouldExit,
		std::deque<char*> *sendQueue, PolicyFunction &policy,
    VehicleState *vehicle, Controller *controller);

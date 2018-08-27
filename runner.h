#pragma once

#include "defines.h"
#include "data.h"
#include "dynamics.h"
#include "diff.h"

void testFeedbackControl(bool *shouldExit,
    char *dataFilename, PolicyFunction &policy,
    std::deque<char*> *sendQueue, VehicleState *vehicle, Controller *controller,
    int numSteps);

int formNextMsg(char *msg);


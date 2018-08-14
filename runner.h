#pragma once

#include "defines.h"
#include "data.h"
#include "dynamics.h"
#include "diff.h"

std::vector<DataPoint> testFeedbackControl(bool *shouldExit,
    char *dataFilename, std::deque<char*> *sendQueue,
    VehicleState *vehicle, Controller *controller,
    int numSteps);

int formNextMsg(char *msg);


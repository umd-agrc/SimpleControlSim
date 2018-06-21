#pragma once

#include <stdlib.h>
#include <unistd.h>
#include <deque>

#include <tiny_dnn/tiny_dnn.h>

#include "defines.h"
#include "data.h"
#include "dynamics.h"
#include "diff.h"
//#include "genann.h"

void *simulate(void *var);

int formNextMsg(char *msg);

int setupSystem(VehicleState *vehicle, Controller *controller, int dataShape[2]);
int setupSystem(VehicleState *vehicle, Controller *controller, FILE *in);
int teardownSystem(VehicleState *vehicle, Controller *controller);

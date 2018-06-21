#pragma once

#include <stdlib.h>
#include <unistd.h>
#include <deque>

#include "defines.h"
#include "data.h"
#include "testDynamics.h"
#include "diff.h"

int setupSystem(VehicleState *vehicle, Controller *controller);
int teardownSystem(VehicleState *vehicle, Controller *controller);

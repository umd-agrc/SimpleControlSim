#ifndef NN_CONTROL_POLICY_EVAL_H_
#define NN_CONTROL_POLICY_EVAL_H_

#include <string>
#include <map>
#include <vector>
#include <algorithm>

#include "defines.h"
#include "policy.h"
#include "runner.h"

void policyUpdate(bool *shouldExit,
		std::deque<char*> *sendQueue, PolicyFunction &policy,
    VehicleState *vehicle, Controller *controller, int horizon);

/* Reward Function
 * percent difference of current state vs desired state scaled to/capped at +-1
 */
void getAdvantage(
    PolicyFunction &policy,
    mx_float horizonDecay = 0.99, mx_float genAdvantageDecay = 0.95);

void getReward(PolicyFunction &policy);

void getProbabilityDistributionRatio(PolicyFunction &policy);

void getMeanKl(PolicyFunction &policy);

void getMeanEnt(PolicyFunction &policy);

void getObjectiveAndGradient(PolicyFunction &policy);

void setupLoss(PolicyFunction &policy, mx_float epsilon = 0.2, mx_float entropyCoeff = 0.05);

#endif // NN_CONTROL_POLICY_EVAL_H_

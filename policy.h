#pragma once

#include <math.h>
#include <tiny_dnn/tiny_dnn.h>
#include <CppAD/cppad/cppad.hpp>

#include "defines.h"
#include "dynamics.h"
#include "diff.h"
#include "runner.h"

void policyUpdate(bool *shouldExit, char *policyNetFile, char *valueNetFile,
    char *trainingDataFile, std::deque<char*> *sendQueue,
    PolicyFunction *policy, VehicleState *vehicle, Controller *controller);

tiny_dnn::vec_t advantageDelta(DataPoint *state, DataPoint *nextState,
    PolicyFunction *policy, double horizonDecay = 0.9);

tiny_dnn::vec_t advantage(int idx, int numSteps, std::vector<DataPoint> *run,
    PolicyFunction *policy, double horizonDecay = 0.9, double genAdvantageDecay = 1.0);

template <class Type>
Type objective(const CppAD::vector<Type> &out,
    const tiny_dnn::vec_t &advantageValues,
    const tiny_dnn::vec_t &in,
    const tiny_dnn::vec_t &value,
    const tiny_dnn::vec_t &valueTarget,
    PolicyFunction *policy,
    double epsilon = 0.2,
    double entropyCoef = 0.1);

void getObjectiveValues(std::vector<DataPoint> *v,
    PolicyFunction *policy);

LossAndGradients getLossAndGradients(std::vector<DataPoint> *v);

void setStateValue(DataPoint *state, PolicyFunction *policy);

void setReward(DataPoint *state);

double clip(double v, double left, double right);

std::vector<double> policyRatio(DataPoint *state);

std::vector<double> stateValue(DataPoint *state);

std::vector<double> policySurrogate(DataPoint *state);

std::vector<double> valueLoss(DataPoint *state);

std::vector<double> entropyBonus(DataPoint *state);

std::vector<double> reward(DataPoint *state, DataPoint *nextState);

std::vector<double> targetValue(DataPoint *state);

std::vector<DataPoint> actor(bool *shouldExit, char *policyNetFile, char *valueNetFile,
    char *trainingDataFile, std::deque<char*> *sendQueue,
    PolicyFunction *policy, VehicleState *vehicle, Controller *controller);

void setValueTrainingData(const std::vector<DataPoint> &v,
    std::vector<tiny_dnn::vec_t> &valueTrainingInput,
    std::vector<tiny_dnn::vec_t> &valueTrainingOutput);

//TODO TDLambda + GAE

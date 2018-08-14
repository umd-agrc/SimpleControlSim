#include "policy.h"

double epsilon;
double c1,c2;
tiny_dnn::adam opt;

using namespace tiny_dnn;

void policyUpdate(bool *shouldExit, char *policyNetFile, char *valueNetFile,
    char *trainingDataFile, std::deque<char*> *sendQueue,
    PolicyFunction *policy, VehicleState *vehicle, Controller *controller) {
  int numActors = 1;
  int totalEpochs = 1;
  std::vector<tiny_dnn::vec_t> valueTrainingInput, valueTrainingOutput;
  std::vector<DataPoint> v;
  LossAndGradients lossAndGradients;
  
  opt.alpha = 0.01;

  //TODO gather clipped loss and -mse+entropy_bonus for each sample
  // Form training minibatch
  for (int i=0; i < numActors; i++) {
    std::vector<DataPoint> vTmp = actor(shouldExit,policyNetFile,valueNetFile,
        trainingDataFile,sendQueue,policy,vehicle,controller);
    v.insert(v.end(),vTmp.begin(),vTmp.end());
  }

  // Get objective values
  getObjectiveValues(&v, policy);

  lossAndGradients = getLossAndGradients(&v);

  policy->policyNet.fit_loss(opt,
                lossAndGradients.loss,
                lossAndGradients.gradients,
                1,[](){},[](){});

  //TODO valueNet fitting
  setValueTrainingData(v,valueTrainingInput,valueTrainingOutput);
  size_t valueTrainingBatchSize = v.size()-1;
  int valueNumEpoch = 10;
  //policy->valueNet.fit<tiny_dnn::mse>(opt,
  policy->valueNet.fit<tiny_dnn::absolute>(opt,
      valueTrainingInput,
      valueTrainingOutput,
      valueTrainingBatchSize,
      valueNumEpoch,
      [](){},[](){});
  
  policy->oldPolicyNet = policy->policyNet;

  if (valueNumEpoch == CAMILA_TYPE) {
    printf("HOORAY BOIIIII!!!");
  }
}

//FIXME
tiny_dnn::vec_t advantageDelta(DataPoint *state, DataPoint *nextState,
    PolicyFunction *policy, double horizonDecay) {
  setReward(state);
  setStateValue(state,policy);
  setStateValue(nextState,policy);
  return state->reward + horizonDecay*nextState->value - state->value;
}

tiny_dnn::vec_t advantage(int idx, int numSteps, std::vector<DataPoint> *run,
    PolicyFunction *policy, double horizonDecay, double genAdvantageDecay) {
  // FIXME what to do if only one step
  if (numSteps == 1) return {0};

  DataPoint d = run->at(idx);
  DataPoint dNext;

  //TODO initialize advantage values to zero
  tiny_dnn::vec_t advantageValues(d.out.size());
  for (int i=idx; i < numSteps-1; i++) {
    DataPoint dNext = run->at(idx+1);
    advantageValues = advantageDelta(&d,&dNext,policy,horizonDecay);
    for (unsigned int j=0; j < advantageValues.size(); j++) {
      advantageValues[j] *= pow(horizonDecay*genAdvantageDecay,i-idx);
    }

    d = dNext;
  }

  return advantageValues;
}

template <class Type>
Type objective(const CppAD::vector<Type> &out,
    const tiny_dnn::vec_t &advantageValues,
    const tiny_dnn::vec_t &in,
    const tiny_dnn::vec_t &value,
    const tiny_dnn::vec_t &valueTarget,
    PolicyFunction *policy,
    double epsilon,
    double entropyCoef) {
  tiny_dnn::vec_t oldMeanPolicy = policy->oldPolicyNet.predict(in);
  CppAD::vector<Type> oldLogp = policy->probabilityDistribution.logp(out);

  tiny_dnn::vec_t meanPolicy = policy->policyNet.predict(in);
  policy->probabilityDistribution.set_mean(meanPolicy);
  CppAD::vector<Type> logp = policy->probabilityDistribution.logp(out);

  CppAD::vector<Type> policyRatio = exp(logp - oldLogp);
  CppAD::vector<Type> surr1 = emult(policyRatio,advantageValues);
  CppAD::vector<Type> surr2 = emult(clip(policyRatio,1-epsilon,1+epsilon),advantageValues);
  CppAD::vector<Type> policySurrogate = min(surr1,surr2);

  Type ent = entropyCoef*policy->probabilityDistribution.meanEntropy<Type>();
  
  return reduce_mean(policySurrogate) + ent + mse(value,valueTarget);
}

void getObjectiveValues(std::vector<DataPoint> *v,
    PolicyFunction *policy) {

  size_t n=v->at(0).out.size();
  size_t m=1;

  for (size_t i = 0; i < v->size(); i++) {
    CppAD::vector<CppAD::AD<double>> outputVar(n);
    outputVar = convertToAdVec<CppAD::AD<double>>(v->at(i).out);

    CppAD::Independent(outputVar);

    CppAD::vector<CppAD::AD<double>> objectiveVar(m);
    objectiveVar[0] = objective(outputVar,
        v->at(i).advantageValues, v->at(i).in,
        v->at(i).value, v->at(i).valueTarget, policy);

    CppAD::ADFun<double> surrogateVars(outputVar,objectiveVar);

    v->at(i).objectiveValue = CppAD::Value(objectiveVar[0]);

    tiny_dnn::vec_t x(n);
    x = v->at(i).out;
    v->at(i).gradientValues = surrogateVars.Jacobian(x);
  }
}

LossAndGradients getLossAndGradients(std::vector<DataPoint> *v) {
  LossAndGradients ret;

  for (auto it=v->begin(); it != v->end(); it++) {
    ret.loss = ret.loss + it->objectiveValue;
    ret.gradients = ret.gradients + it->gradientValues;
  }
  ret.loss = ret.loss / v->size();
  ret.gradients = ret.gradients / v->size();
}

void setStateValue(DataPoint *state, PolicyFunction *policy) {
  //TODO how to compute this? Do we need the value network?
  state->value = policy->valueNet.predict(vec_t_concat(state->out,state->in));
}

//TODO
void setReward(DataPoint *state) {
  state->reward = 0; 
}

std::vector<DataPoint> actor(bool *shouldExit, char *policyNetFile, char *valueNetFile,
    char *trainingDataFile, std::deque<char*> *sendQueue,
    PolicyFunction *policy, VehicleState *vehicle, Controller *controller) {
  int numSteps = 1000;
  
  std::vector<DataPoint> v =
    testFeedbackControl(shouldExit,NULL,sendQueue,vehicle,controller,numSteps);

  //FIXME should go to full episode termination
  for (int i=0; i < numSteps-1; i++) {
    v[i].advantageValues = advantage(i, numSteps, &v, policy);
    v[i].valueTarget = v[i].advantageValues + v[i].value;
  }
  return v;
}

void setValueTrainingData(const std::vector<DataPoint> &v,
    std::vector<tiny_dnn::vec_t> &valueTrainingInput,
    std::vector<tiny_dnn::vec_t> &valueTrainingOutput) {
  size_t k = v.size();
  valueTrainingInput.resize(k);
  valueTrainingOutput.resize(k);
  for (size_t i = 0; i < k; i++) {
    valueTrainingInput[i] = vec_t_concat(v[i].out,v[i].in);
    valueTrainingOutput[i] = v[i].valueTarget;
  }
}

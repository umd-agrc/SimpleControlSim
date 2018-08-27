#include "policy_eval.h"

using namespace mxnet::cpp;

void policyUpdate(bool *shouldExit,
		std::deque<char*> *sendQueue, PolicyFunction &policy,
    VehicleState *vehicle, Controller *controller, int horizon) {
  testFeedbackControl(shouldExit,NULL,policy,sendQueue,vehicle,controller,horizon);

  /*
  getAdvantage(policy);
  getProbabilityDistributionRatio(policy);
  getMeanKl(policy);
  getMeanEnt(policy);
  getObjectiveAndGradient(policy);

  policy.update(policy.lossExec->grad_arrays,policy.valueExec->grad_arrays);
  */
}

void getAdvantage(
    PolicyFunction &policy,
    mx_float horizonDecay, mx_float genAdvantageDecay) {
  getReward(policy);

  policy.valueArgs["valuex"] = policy.trajSegment["observationAndAction"];
  policy.valueExec->Forward(true);
  policy.trajSegment["vpred"] = policy.valueExec->outputs[0];
  auto shape = policy.valueExec->outputs[0].GetShape();
  std::vector<mx_float> advantage;
  NDArray delta, vpred = policy.valueExec->outputs[0];
  mx_float gaeLamPrev;
  // Handle final element
  delta = policy.trajSegment["reward"].Slice(shape[0]-1,shape[0])
          - vpred.Slice(shape[0]-1,shape[0]);
  gaeLamPrev = *(delta.GetData());
  advantage.push_back(gaeLamPrev);
  for (int i = shape[0]-1; i >= 0; i++) {
    delta = policy.trajSegment["reward"].Slice(i-1,i)
            + vpred.Slice(i, i+1)*horizonDecay
            - vpred.Slice(i-1,i);
    gaeLamPrev = *((delta + horizonDecay*genAdvantageDecay*gaeLamPrev).GetData());
    advantage.push_back(gaeLamPrev);
  }
  std::reverse(advantage.begin(),advantage.end());
  policy.trajSegment["advantage"] = NDArray(advantage,
                                     Shape(shape[0],1),
                                     Context::cpu());
}

void getReward(PolicyFunction &policy) {
  auto shape = policy.trajSegment["observation"].GetShape();
  std::vector<mx_float> reward(shape[0]);
  for(size_t i = 0; i < shape[0]; i++) {
  NDArray currObs = policy.trajSegment["observation"].Slice(i,i+1);
    mx_float avgPercentDiff = 0;
    for(size_t j = 0; j < NUM_STATES; j++) {
      avgPercentDiff +=
        1 - abs((currObs.At(0,j)-currObs.At(0,j+NUM_STATES))
                /(currObs.At(0,j)+currObs.At(0,j+NUM_STATES))/2);
    }
    avgPercentDiff /= NUM_STATES;
    reward.push_back(clip(avgPercentDiff,-1,1));
  }
  policy.trajSegment["reward"] = NDArray(reward,Shape(1,shape[0]),Context::cpu());
}

void getProbabilityDistributionRatio(PolicyFunction &policy) {
  policy.trajSegment["ratio"] =
    exp(policy.probabilityDistribution.logp(policy.trajSegment)
        - policy.probabilityDistribution.logp(policy.oldTrajSegment));
}

void getMeanKl(PolicyFunction &policy) {
  policy.trajSegment["meanKl"] =
    mean(policy.probabilityDistribution.kl(policy.trajSegment,policy.oldTrajSegment));
}

void getMeanEnt(PolicyFunction &policy) {
  policy.trajSegment["meanEnt"] =
    mean(policy.probabilityDistribution.entropy(policy.trajSegment));
}

void getObjectiveAndGradient(PolicyFunction &policy) {
  policy.lossExec->Forward(true);
  policy.valueExec->Backward();
  std::vector<mx_float> tmp(policy.lossExec->outputs.size());
  for (size_t i=0; i < tmp.size(); i++) {
    tmp[i] = *policy.lossExec->outputs[i].GetData();
  }
  policy.trajSegment["policyLoss"] = NDArray(tmp,Shape(1,tmp.size()),Context::cpu());

  policy.lossExec->Backward(policy.lossExec->outputs);
}

//TODO For evaluation inputs:
//     keep track of pdmean (output of policyNet)
//     keep track of stochastic action
void setupLoss(PolicyFunction &policy, mx_float epsilon, mx_float entropyCoeff) {
  Symbol advantage = Symbol::Variable("advantage");
  Symbol ratio = Symbol::Variable("ratio");
  Symbol meanKl = Symbol::Variable("meanKl");
  Symbol meanEnt = Symbol::Variable("meanEnt");
  Symbol vpred = Symbol::Variable("vpred");
  Symbol vref = advantage + vpred;
  Symbol vfLoss = mean(square(vpred - vref));
  Symbol surr1 = ratio*advantage;
  Symbol surr2 = clip(ratio,1-epsilon,1+epsilon);
  Symbol polSurr = mean(min(surr1,surr2));
  //FIXME polEntPen causes things to break for some reason
  //Symbol polEntPen = -entropyCoeff*meanEnt;
  //policy.loss = polSurr + vfLoss + polEntPen
  policy.loss = polSurr + vfLoss;
  
  policy.trajSegment["action"] = NDArray(Shape(1,NUM_INPUTS),Context::cpu(),true);
  policy.trajSegment["advantage"] = NDArray(Shape(1,1),Context::cpu(),true);
  policy.trajSegment["ratio"] = NDArray(Shape(1,1),Context::cpu(),true);
  policy.trajSegment["meanKl"] = NDArray(Shape(1,1),Context::cpu(),true);
  policy.trajSegment["meanEnt"] = NDArray(Shape(1,1),Context::cpu(),true);
  policy.trajSegment["vpred"] = NDArray(Shape(1,1),Context::cpu(),true);

  policy.lossExec = policy.loss.SimpleBind(Context::cpu(),policy.trajSegment); 
}

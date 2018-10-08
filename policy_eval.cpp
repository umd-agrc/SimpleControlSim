#include "policy_eval.h"

using namespace mxnet::cpp;

#ifdef DEBUG
#include <chrono>
using namespace std::chrono;
static high_resolution_clock::time_point t1,t2;
#endif

void policyUpdate(bool *shouldExit,
		std::deque<char*> *sendQueue, PolicyFunction &policy,
    VehicleState *vehicle, Controller *controller, int horizon) {
  log("Running policy update");
  testFeedbackControl(shouldExit,NULL,&policy,NULL,sendQueue,vehicle,controller,horizon);
  getAdvantage(policy);
  getObjectiveAndGradient(policy);

  auto lossGrads = policy.lossExec->grad_dict();

  std::vector<NDArray> meanGrads = {lossGrads["action"]};
  //std::cout << "lossgrads " << meanGrads[0] << std::endl;
#ifdef DEBUG
  t1 = high_resolution_clock::now();
#endif
  policy.policyExec->Backward(meanGrads);
  NDArray::WaitAll();
#ifdef DEBUG
  t2 = high_resolution_clock::now();
  policy.executionTimesVec["policyBatchBackwardPass"].push_back(
      duration_cast<microseconds>(t2-t1).count());
#endif
  policy.update(policy.policyExec->grad_arrays,policy.valueExec->grad_arrays);
}

void getAdvantage(
    PolicyFunction &policy,
    mx_float horizonDecay, mx_float genAdvantageDecay) {
  getReward(policy);

  policy.valueArgs["valuex"] = policy.trajSegment["observationAndAction"];
  policy.rebindValue();
  NDArray::WaitAll();
#ifdef DEBUG
  t1 = high_resolution_clock::now();
#endif
  policy.valueExec->Forward(true);
  NDArray::WaitAll();
#ifdef DEBUG
  t2 = high_resolution_clock::now();
  policy.executionTimesVec["valueBatchForwardPass"].push_back(
      duration_cast<microseconds>(t2-t1).count());
#endif
  policy.trajSegment["vpred"] = policy.valueExec->outputs[0];
  NDArray::WaitAll();
  //std::cout << "vpred " << policy.trajSegment["vpred"] << std::endl;
  auto shape = policy.valueExec->outputs[0].GetShape();
  NDArray advantage(shape,Context::cpu(),false);
  NDArray delta, vpred = policy.valueExec->outputs[0];
  NDArray::WaitAll();
  mx_float gaeLamPrev;
  // Handle final element
  delta = policy.trajSegment["reward"].Slice(shape[0]-1,shape[0])
          - vpred.Slice(shape[0]-1,shape[0]);
  NDArray::WaitAll();
  auto d1 = policy.trajSegment["reward"].Slice(shape[0]-1,shape[0]); 
  auto d2 = vpred.Slice(shape[0]-1,shape[0]);
  gaeLamPrev = delta.At(0,0);
  advantage.SetData(shape[0]-1,gaeLamPrev);
  NDArray::WaitAll();
  for (int i = shape[0]-2; i >= 0; i--) {
    delta = policy.trajSegment["reward"].Slice(i,i+1)
            + vpred.Slice(i+1, i+2)*horizonDecay
            - vpred.Slice(i,i+1);
    NDArray::WaitAll();
    gaeLamPrev = delta.At(0,0) + horizonDecay*genAdvantageDecay*gaeLamPrev;
    NDArray::WaitAll();
    advantage.SetData(i,gaeLamPrev);
    NDArray::WaitAll();
    /*
    std::cout << "gaelamprev " << i << " " << horizonDecay*genAdvantageDecay*gaeLamPrev << std::endl;
    std::cout << "delta " << i << " " << delta << " vs " << delta.At(0,0) << std::endl;
    std::cout << "advantage " << i << " " << gaeLamPrev << std::endl;
    */
  }
  policy.trajSegment["advantage"] = advantage;
  NDArray::WaitAll();
}

void getReward(PolicyFunction &policy) {
  auto shape = policy.trajSegment["observation"].GetShape();
  NDArray reward(Shape(shape[0],1),Context::cpu(),false);
  NDArray initObs = policy.trajSegment["observation"].Slice(0,1);
  NDArray::WaitAll();
  for(size_t i = 0; i < shape[0]; i++) {
    NDArray currObs = policy.trajSegment["observation"].Slice(i,i+1);
    mx_float avgPercentDiff = 0;
    NDArray::WaitAll();
    for(size_t j = 0; j < NUM_STATES; j++) {
      avgPercentDiff +=
        1 - abs((currObs.At(0,j)-currObs.At(0,j+NUM_STATES))
                /fmax(fabs(initObs.At(0,j)-initObs.At(0,j+NUM_STATES)),0.001));
      NDArray::WaitAll();
    }
    avgPercentDiff /= NUM_STATES;
    NDArray::WaitAll();
    reward.SetData(i,clip(avgPercentDiff,-1,1));
    NDArray::WaitAll();
  }
  policy.trajSegment["reward"] = reward;
  NDArray::WaitAll();
}

// Get loss and gradients for policy function and value function
void getObjectiveAndGradient(PolicyFunction &policy) {
  policy.rebindLoss();

#ifdef DEBUG
  t1 = high_resolution_clock::now();
#endif
  policy.lossExec->Forward(true);
  NDArray::WaitAll();
#ifdef DEBUG
  t2 = high_resolution_clock::now();
  policy.executionTimesVec["lossBatchForwardPass"].push_back(
      duration_cast<microseconds>(t2-t1).count());
#endif

  policy.trajSegment["policyLoss"] = policy.lossExec->outputs[0];
  NDArray::WaitAll();
  //std::cout << "meanctl " << policy.trajSegment["meanCtl"] << std::endl;
  std::cout << "action " << policy.trajSegment["action"] << std::endl;
  //std::cout << "std " << policy.trajSegment["std"] << std::endl;
  std::cout << "observation " << policy.trajSegment["observation"] << std::endl;
  std::cout << "policy loss " << policy.trajSegment["policyLoss"] << std::endl;

#ifdef DEBUG
  t1 = high_resolution_clock::now();
#endif
  policy.lossExec->Backward(policy.lossExec->outputs);
  NDArray::WaitAll();
#ifdef DEBUG
  t2 = high_resolution_clock::now();
  policy.executionTimesVec["lossBatchBackwardPass"].push_back(
      duration_cast<microseconds>(t2-t1).count());
#endif

  std::vector<NDArray> vtarg = {policy.trajSegment["advantage"] + policy.trajSegment["vpred"]};
  NDArray::WaitAll();
#ifdef DEBUG
  t1 = high_resolution_clock::now();
#endif
  policy.valueExec->Backward(vtarg);
  NDArray::WaitAll();
#ifdef DEBUG
  t2 = high_resolution_clock::now();
  policy.executionTimesVec["valueBatchBackwardPass"].push_back(
      duration_cast<microseconds>(t2-t1).count());
#endif
}

void setupLoss(PolicyFunction &policy, mx_float epsilon, mx_float entropyCoeff) {
  Symbol advantage = Symbol::Variable("advantage");
  Symbol vpred = Symbol::Variable("vpred");
  Symbol meanCtl = Symbol::Variable("meanCtl");
  Symbol std = Symbol::Variable("std");
  Symbol action = Symbol::Variable("action");
  Symbol oldMeanCtl = Symbol::Variable("oldMeanCtl");
  Symbol oldStd = Symbol::Variable("oldStd");
  Symbol oldAction = Symbol::Variable("oldAction");

  Symbol logstd = log(std);
  Symbol oldLogstd = log(oldStd);
  Symbol ratio =
    exp(policy.probabilityDistribution.logp(meanCtl,std,logstd,action)
    		- policy.probabilityDistribution.logp(oldMeanCtl,oldStd,oldLogstd,oldAction));
  Symbol meanKl = mean(policy.probabilityDistribution.kl(meanCtl,std,logstd,
                                                         oldMeanCtl,oldStd,oldLogstd),
                       dmlc::optional<Shape>(Shape(0)));
  Symbol meanEnt = mean(policy.probabilityDistribution.entropy(logstd),
                        dmlc::optional<Shape>(Shape(0)));
  Symbol vref = advantage + vpred;
  Symbol vfLoss = mean(square(vpred - vref),dmlc::optional<Shape>(Shape(0)));
  Symbol surr1 = reshape_like(ratio,advantage)*advantage;
  Symbol surr2 = reshape_like(clip(ratio,1-epsilon,1+epsilon),surr1)*advantage;
  Symbol polSurr = mean(min(surr1,surr2),dmlc::optional<Shape>(Shape(0)));
  Symbol polEntPen = negative(meanEnt*entropyCoeff);
  //policy.loss= polSurr + vfLoss + polEntPen; 
  //policy.loss= polSurr;
  policy.loss = polSurr;
  //policy.loss = meanKl;
  
  policy.trajSegment["advantage"] = NDArray(Shape(1,1),Context::cpu(),true);
  policy.trajSegment["vpred"] = NDArray(Shape(1,1),Context::cpu(),true);
  policy.trajSegment["meanCtl"] = NDArray(Shape(1,NUM_INPUTS),Context::cpu(),true);
  policy.trajSegment["std"] = NDArray(Shape(1,NUM_INPUTS),Context::cpu(),true);
  policy.trajSegment["action"] = NDArray(Shape(1,NUM_INPUTS),Context::cpu(),true);
  policy.trajSegment["oldMeanCtl"] = NDArray(Shape(1,NUM_INPUTS),Context::cpu(),true);
  policy.trajSegment["oldStd"] = NDArray(Shape(1,NUM_INPUTS),Context::cpu(),true);
  policy.trajSegment["oldAction"] = NDArray(Shape(1,NUM_INPUTS),Context::cpu(),true);

  policy.lossExec = policy.loss.SimpleBind(Context::cpu(),policy.trajSegment); 
  policy.lossBound = true;
}

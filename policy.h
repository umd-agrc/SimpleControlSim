#ifndef NN_CONTROL_POLICY_H_
#define NN_CONTROL_POLICY_H_

#include <vector>

#include "defines.h"
#include "utils.h"
#include "probability_distributions.h"

class PolicyFunction {
public:
  mxnet::cpp::Symbol policyNet;
  mxnet::cpp::Symbol valueNet;
  mxnet::cpp::Symbol loss;
  DiagGaussianPd probabilityDistribution;
  mx_float stdDefaultValue;

  bool lossBound = false,
       oldPolicyBound = false;

  mxnet::cpp::Optimizer *policyOpt;
  mxnet::cpp::Optimizer *valueOpt;

  std::map<std::string,mxnet::cpp::NDArray> policyArgs;
  std::map<std::string,mxnet::cpp::NDArray> oldPolicyArgs;
  std::map<std::string,mxnet::cpp::NDArray> valueArgs;
  std::map<std::string,mxnet::cpp::NDArray> trajSegment;
  std::map<std::string,mxnet::cpp::NDArray> oldTrajSegment;;

  // Filename strings
  std::string policyNetString = "sym/policyNet";
  std::string valueNetString = "sym/valueNet";
  std::string lossString = "sym/loss";
  std::string policyArgsString = "arr/policyArgs";
  std::string oldPolicyArgsString = "arr/oldPolicyArgs";
  std::string valueArgsString = "arr/valueArgs";
  std::string trajSegmentString = "arr/trajSegment";
  std::string oldTrajSegmentString = "arr/oldTrajSegment";

  mxnet::cpp::NDArray baseActions;

#ifdef DEBUG
  std::string executionTimesString = "dbg/executionTimes";
  std::map<std::string,mxnet::cpp::NDArray> executionTimes;
  std::map<std::string,std::vector<mx_float>> executionTimesVec;
#endif

  mxnet::cpp::Executor *policyExec;
  mxnet::cpp::Executor *oldPolicyExec;
  mxnet::cpp::Executor *valueExec;
  mxnet::cpp::Executor *lossExec;

  PolicyFunction();
  ~PolicyFunction();

  //std::vector<mxnet::cpp::NDArray> act(mxnet::cpp::NDArray observations);
  void update(std::vector<mxnet::cpp::NDArray> policyGradients,
    std::vector<mxnet::cpp::NDArray> valueGradients);

  mxnet::cpp::NDArray getRand(mxnet::cpp::Shape shape);
  mxnet::cpp::NDArray getStd(mxnet::cpp::Shape shape);
  mxnet::cpp::NDArray sample();

  void rebindPolicy();
  void rebindOldPolicy();
  void rebindValue();
  void rebindLoss();

  void saveArr(std::string filenamePrefix,std::string filenameSuffix);
  void loadArr(std::string filenamePrefix,std::string filenameSuffix);
  void saveSym(std::string filenamePrefix,std::string filenameSuffix);
  void loadSym(std::string filenamePrefix,std::string filenameSuffix);

  void teardown();
};

#endif

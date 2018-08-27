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
  mxnet::cpp::Symbol logstd;
  mxnet::cpp::Symbol loss;
  DiagGaussianPd probabilityDistribution;
  mx_float stdDefaultValue;

  mxnet::cpp::Optimizer *policyOpt;
  mxnet::cpp::Optimizer *valueOpt;

  std::map<std::string, mxnet::cpp::NDArray> policyArgs;
  std::map<std::string, mxnet::cpp::NDArray> oldPolicyArgs;
  std::map<std::string, mxnet::cpp::NDArray> valueArgs;
  std::map<std::string,mxnet::cpp::NDArray> trajSegment;
  std::map<std::string,mxnet::cpp::NDArray> oldTrajSegment;;
  mxnet::cpp::NDArray baseActions;

  mxnet::cpp::Executor *policyExec;
  mxnet::cpp::Executor *oldPolicyExec;
  mxnet::cpp::Executor *valueExec;
  mxnet::cpp::Executor *lossExec;

  PolicyFunction();

  std::vector<mxnet::cpp::NDArray> act(mxnet::cpp::NDArray observations);
  void update(std::vector<mxnet::cpp::NDArray> policyGradients,
    std::vector<mxnet::cpp::NDArray> valueGradients);

  mxnet::cpp::NDArray getRand(mxnet::cpp::Shape shape);
  mxnet::cpp::NDArray getStd(mxnet::cpp::Shape shape);
  mxnet::cpp::NDArray sample();

  void teardown();
};

#endif

#include "policy.h"

using namespace mxnet::cpp;

PolicyFunction::PolicyFunction() {
  const int policyInputSize = 2*NUM_STATES;
  const int policyOutputSize = NUM_INPUTS;
  const int valueInputSize = 2*NUM_STATES+NUM_INPUTS;
  const int valueOutputSize = 1;

	const std::vector<int> policyLayers{100,100,policyOutputSize};
	const std::vector<int> valueLayers{100,100,valueOutputSize};

  const int batchSize = 1;
  const int maxEpoch = 1;

  // Set up probability distributions
  stdDefaultValue = 0.1;

  policyNet = mlp("policy",policyLayers);
  valueNet = mlp("value",valueLayers);

  policyArgs["policyx"] = NDArray(Shape(batchSize, policyInputSize), Context::cpu(), true);
  policyArgs["policyy"] = NDArray(Shape(batchSize, policyOutputSize), Context::cpu(), true);

  valueArgs["valuex"] = NDArray(Shape(batchSize, valueInputSize), Context::cpu(), true);
  valueArgs["valuey"] = NDArray(Shape(batchSize, valueOutputSize), Context::cpu(), true);

  policyNet.InferArgsMap(Context::cpu(), &policyArgs, policyArgs);
  valueNet.InferArgsMap(Context::cpu(), &valueArgs, valueArgs);

  // Initialize all parameters with uniform distribution U(-0.01, 0.01)
  auto initializer = Uniform(0.01);
  for (auto& arg : policyArgs) {
    // arg.first is parameter name, and arg.second is the value
    initializer(arg.first, &arg.second);
  }

  // Initialize all parameters with uniform distribution U(-0.01, 0.01)
  for (auto& arg : valueArgs) {
    // arg.first is parameter name, and arg.second is the value
    initializer(arg.first, &arg.second);
  }

  policyOpt = OptimizerRegistry::Find("adam");
	policyOpt->SetParam("lr", 0.01);
  valueOpt = OptimizerRegistry::Find("adam");
	valueOpt->SetParam("lr", 0.01);

  policyExec = policyNet.SimpleBind(Context::cpu(),policyArgs); 
  valueExec = valueNet.SimpleBind(Context::cpu(),valueArgs); 
}

std::vector<NDArray> PolicyFunction::act(NDArray observations) {
  std::vector<NDArray> ret;
  //TODO bind observations to policyExec
  policyExec->Forward(false);
  ret.push_back(policyExec->outputs[0]);
  //TODO bind observations + policy outputs to valueExec
  valueExec->Forward(false);
  ret.push_back(valueExec->outputs[0]);
  return ret;
}

void PolicyFunction::update(std::vector<NDArray> policyGradients,
    std::vector<NDArray> valueGradients) {

  // Keep copy of old policy
  oldPolicyExec->arg_arrays = policyExec->arg_arrays;

  auto policyArgNames = policyNet.ListArguments();
  for (size_t i = 0; i < policyArgNames.size(); ++i) {
    if (policyArgNames[i] == "policyx" || policyArgNames[i] == "policyy") continue;
    policyOpt->Update(i, policyExec->arg_arrays[i], policyGradients[i]);
  }

  auto valueArgNames = valueNet.ListArguments();
  for (size_t i = 0; i < valueArgNames.size(); ++i) {
    if (valueArgNames[i] == "valuex" || valueArgNames[i] == "valuey") continue;
    valueOpt->Update(i, valueExec->arg_arrays[i], valueGradients[i]);
  }
}

NDArray PolicyFunction::getRand(Shape shape) {
  NDArray ret = NDArray(shape,Context::cpu(),true);
  NDArray::SampleGaussian(0,1,&ret);
  return ret;
}

NDArray PolicyFunction::getStd(Shape shape) {
  NDArray ret = NDArray(shape,Context::cpu(),true);
  ret = stdDefaultValue;
  return ret;
}

//TODO need to evaluate mean
//TODO sample option old or new
NDArray PolicyFunction::sample() {
  std::map<std::string,NDArray> policy_dict = policyExec->arg_dict();
  return policy_dict["policyy"] + getStd(Shape(1,NUM_INPUTS));
}

void PolicyFunction::teardown() {
  delete policyExec;
  delete oldPolicyExec;
  delete valueExec;
  delete lossExec;
}

#include "policy.h"

using namespace mxnet::cpp;

PolicyFunction::PolicyFunction() {
  log("Constructing policy function");
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
    if (arg.first == "policyx" || arg.first == "policyy") continue;
    initializer(arg.first, &arg.second);
  }

  // Initialize all parameters with uniform distribution U(-0.01, 0.01)
  for (auto& arg : valueArgs) {
    // arg.first is parameter name, and arg.second is the value
    if (arg.first == "valuex" || arg.first == "valuey") continue;
    initializer(arg.first, &arg.second);
  }

  policyOpt = OptimizerRegistry::Find("adam");
	policyOpt->SetParam("lr", 0.01);
  valueOpt = OptimizerRegistry::Find("adam");
	valueOpt->SetParam("lr", 0.01);

  policyExec = policyNet.SimpleBind(Context::cpu(),policyArgs); 
  valueExec = valueNet.SimpleBind(Context::cpu(),valueArgs); 

}

PolicyFunction::~PolicyFunction() {
  log("Deconstructing policy function");
  teardown();
}

/*
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
*/

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

NDArray PolicyFunction::sample() {
  std::map<std::string,NDArray> policy_dict = policyExec->arg_dict();
  return policy_dict["policyy"] + getStd(Shape(1,NUM_INPUTS));
}

void PolicyFunction::rebindPolicy() {
  policyNet.InferArgsMap(Context::cpu(),&policyArgs,policyArgs);
  policyExec = policyNet.SimpleBind(Context::cpu(),policyArgs);
}

void PolicyFunction::rebindOldPolicy() {
  oldPolicyBound = true;
  policyNet.InferArgsMap(Context::cpu(),&oldPolicyArgs,oldPolicyArgs);
  oldPolicyExec = policyNet.SimpleBind(Context::cpu(),oldPolicyArgs);
}

void PolicyFunction::rebindValue() {
  valueNet.InferArgsMap(Context::cpu(),&valueArgs,valueArgs);
  valueExec = valueNet.SimpleBind(Context::cpu(),valueArgs);
}

void PolicyFunction::rebindLoss() {
  lossBound = true;
  lossExec = loss.SimpleBind(Context::cpu(),trajSegment);
}

void PolicyFunction::saveArr(std::string filenamePrefix,std::string filenameSuffix) {
  NDArray::Save(filenamePrefix+policyArgsString+filenameSuffix,policyArgs);
  NDArray::Save(filenamePrefix+oldPolicyArgsString+filenameSuffix,oldPolicyArgs);
  NDArray::Save(filenamePrefix+valueArgsString+filenameSuffix,valueArgs);
  NDArray::Save(filenamePrefix+trajSegmentString+filenameSuffix,trajSegment);
  NDArray::Save(filenamePrefix+oldTrajSegmentString+filenameSuffix,oldTrajSegment);
}

void PolicyFunction::loadArr(std::string filenamePrefix,std::string filenameSuffix) {
  NDArray::Load(filenamePrefix+policyArgsString+filenameSuffix,nullptr,&policyArgs);
  NDArray::Load(filenamePrefix+oldPolicyArgsString+filenameSuffix,nullptr,&oldPolicyArgs);
  NDArray::Load(filenamePrefix+valueArgsString+filenameSuffix,nullptr,&valueArgs);
  NDArray::Load(filenamePrefix+trajSegmentString+filenameSuffix,nullptr,&trajSegment);
  NDArray::Load(filenamePrefix+oldTrajSegmentString+filenameSuffix,nullptr,&oldTrajSegment);
}

void PolicyFunction::saveSym(std::string filenamePrefix,std::string filenameSuffix) {
  policyNet.Save(filenamePrefix+policyNetString+filenameSuffix);
  valueNet.Save(filenamePrefix+valueNetString+filenameSuffix);
  loss.Save(filenamePrefix+lossString+filenameSuffix);
}

void PolicyFunction::loadSym(std::string filenamePrefix,std::string filenameSuffix) {
  policyNet = Symbol::Load(filenamePrefix+policyNetString+filenameSuffix);
  valueNet = Symbol::Load(filenamePrefix+valueNetString+filenameSuffix);
  loss = Symbol::Load(filenamePrefix+lossString+filenameSuffix);
}

void PolicyFunction::teardown() {
  delete policyExec;
  if (oldPolicyBound) {
    delete oldPolicyExec;
    oldPolicyBound = false;
  }
  delete valueExec;
  if (lossBound) {
    delete lossExec;
    lossBound = false;
  }
}

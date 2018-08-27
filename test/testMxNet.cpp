#include <iostream>
#include <vector>
#include <chrono>
#include <mxnet-cpp/MxNetCpp.h>

#include "utils.h"

using namespace mxnet::cpp;

Symbol test_mlp(const std::vector<int> &layers) {
	auto x = Symbol::Variable("X");
	auto label = Symbol::Variable("label");
	
	std::vector<Symbol> weights(layers.size());
	std::vector<Symbol> biases(layers.size());
	std::vector<Symbol> outputs(layers.size());

	for(size_t i=0; i < layers.size(); i++) {
		weights[i] = Symbol::Variable("w" + std::to_string(i));
		biases[i] = Symbol::Variable("b" + std::to_string(i));
		Symbol fc = FullyConnected(
			i == 0? x : outputs[i-1], // data
			weights[i],	
			biases[i],	
			layers[i]);
		//TODO ensure fully connected
    outputs[i] = i == layers.size()-1 ? fc : Activation(fc, ActivationActType::kRelu);
	}
	//TODO change
	Symbol out = LinearRegressionOutput(outputs.back(), label);
  return out;
}

int main(int argc, char **argv) {
	
  std::cout << "TEST: gradient calculation" << std::endl;

  size_t numSamples = 2;
	std::map<std::string, NDArray> args;
  args["a"] = NDArray(Shape(2,2), Context::cpu(), true);
  args["b"] = NDArray(Shape(2,2), Context::cpu(), true);
  //args["c"] = NDArray(Shape(2,2), Context::cpu(), true);

  std::vector<mx_float> x(4);
  for (int i=0; i < 4; i++) x[i] = i;
  args["a"] = NDArray(x, Shape(2,2), Context::cpu());
  args["b"] = NDArray({0,0,0,0}, Shape(2,2), Context::cpu());
  
  Symbol a = Symbol::Variable("a");
  Symbol b = Symbol::Variable("b");
  Symbol c = max(2*a*a + b,b);
  //Symbol c = sum(a,dmlc::optional<Shape>(Shape(1)));

  auto *exec = c.SimpleBind(Context::cpu(),args); 

  exec->Forward(true);

  std::vector<NDArray> grads = exec->outputs;
  exec->Backward(grads);
  
	c.InferArgsMap(Context::cpu(), &args, args);
  auto arg_names = c.ListArguments();
  for (size_t i = 0; i < arg_names.size(); i++) {
    std::cout << arg_names[i] << "(" << exec->arg_arrays[i].GetShape()[0] << ","
              << exec->arg_arrays[i].GetShape()[1] << ") = " << exec->arg_arrays[i]
              << " ; df/d" << arg_names[i] << " = "
              << exec->grad_arrays[i] << std::endl;
  }
  std::cout << "outputs: " << exec->outputs[0] << std::endl;

  std::cout << std::endl;
  std::cout << "TEST: updating parameters and keeping copy" << std::endl;

	const std::vector<int> layers{100,100,1};
	const int max_epoch = 2;
	const float learning_rate = 0.1;

	auto net = test_mlp(layers);
	std::map<std::string, NDArray> netArgs;
  Context ctx = Context::cpu();
	netArgs["X"] = NDArray({1,2,10,4,11,43}, Shape(2, 3), ctx);
	netArgs["label"] = NDArray({1000,900},Shape(2), ctx);
	net.InferArgsMap(ctx, &netArgs, netArgs);

	auto initializer = Uniform(0.01);
	for (auto &arg : netArgs) {
		initializer(arg.first, &arg.second);
	}

	Optimizer *netOpt = OptimizerRegistry::Find("adam");
	netOpt->SetParam("lr", learning_rate);

	auto *netExec = net.SimpleBind(ctx, netArgs);
	auto netArgNames = net.ListArguments();

  //Symbol net2 = net.Copy();
  //std::map<std::string,NDArray> net2Args;
  //copyNDArrayMap(net2Args,netArgs);
	//auto *net2Exec = net2.SimpleBind(ctx, net2Args);
	//auto net2ArgNames = net2.ListArguments();
  auto argArrsInit = netExec->arg_arrays;

  for (int iter = 0; iter < max_epoch; ++iter) {
    // Compute gradients
    netExec->Forward(true);
    netExec->Backward();
    // Update parameters
    for (size_t i = 0; i < netArgNames.size(); ++i) {
      if (netArgNames[i] == "X" || netArgNames[i] == "label") continue;
      netOpt->Update(i, netExec->arg_arrays[i], netExec->grad_arrays[i]);
    }

    std::cout << "output: " << netExec->outputs[0] << std::endl;
    std::cout << std::endl;
  }

	netArgs["X"] = NDArray({4,11,43}, Shape(1, 3), ctx);
	netArgs["label"] = NDArray({900},Shape(1), ctx);
	net.InferArgsMap(ctx, &netArgs, netArgs);
	netExec = net.SimpleBind(ctx, netArgs);
  netExec->Forward(false);
  std::cout << "output: " << netExec->outputs[0] << std::endl;

  std::cout << std::endl;
  std::cout << "TEST: NDArray multiplication" << std::endl;
  auto a1 = NDArray({2,3,4,3,2,4,3,4,2}, Shape(3,3), ctx);
  auto a2 = NDArray({10,10,10,20,20,20,30,30,30}, Shape(3,3), ctx);

  auto a3 = a1*a2;

  std::cout << "NDArray mult: " << a3 << std::endl;
  std::cout << std::endl;

  std::cout << "TEST: NDArray assignment" << std::endl;
  auto a4 = NDArray(Shape(2,2),ctx,true);
  a4 = 3.4;
  std::cout << "NDArray assignment: " << a4 << std::endl;
  std::cout << std::endl;

  std::cout << "TEST: NDArray reduce sum" << std::endl;
  auto a5 = NDArray({1,2,2,2,3,6,32,12,44,1,2,3,3,54,6,4},
                    Shape(2,2,4),
                    Context::cpu());
  auto a6 = sum(a5,1);
  std::cout << a5 << std::endl;
  std::cout << a6 << std::endl;
  std::cout << std::endl;

  std::cout << "TEST: NDArray log" << std::endl;
  auto a7 = log(a5);
  std::cout << a7 << std::endl;
  std::cout << std::endl;

  //TODO dot product & matrix multiplication
  //FIXME NDArray * Symbol
  
  NDArray::WaitAll();

	delete exec;
	delete netExec;
	MXNotifyShutdown();
	return 0;
}

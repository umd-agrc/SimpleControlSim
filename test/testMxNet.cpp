#include <iostream>
#include <vector>
#include <chrono>
#include <mxnet-cpp/MxNetCpp.h>

#include "testMxNet.h"

using namespace mxnet::cpp;

Context ctx = Context::cpu();

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
  /*
  testGrad();
  testNetUpdate();
  */
  testFunction("Net reshape",&testNetReshape);
  /*
  testNDArrayBasics();
  testSimpleArithmetic();
  testNDArrayReduce();
  testExp();
  testNDArrayAssign();
  testNDArrayConcat();
  testNDArrayInit();
  testNDArrayMax();
  testNDArrayCopy();
  testFunction("NDArray IO",&testNDArrayIo);
  testSymbolReduce();
  testSymbolIo();
  testLogging();
  */

  //FIXME NDArray * Symbol

	MXNotifyShutdown();
	return 0;
}

int testGrad() {
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
  std::cout << "------------------" << std::endl;
  std::cout << std::endl;
  
  NDArray::WaitAll();
	delete exec;
  return TEST_SUCCESS;
}

int testNetUpdate() {
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

	netArgs["X"] = NDArray({1,2,10,4,11,43,33,100,12}, Shape(3, 3), ctx);
	netArgs["label"] = NDArray({1000,900,700},Shape(3), ctx);
	net.InferArgsMap(ctx, &netArgs, netArgs);
	netExec = net.SimpleBind(ctx, netArgs);
  netExec->Forward(false);
  std::cout << "output: " << netExec->outputs[0] << std::endl;

  std::cout << "------------------" << std::endl;
  std::cout << std::endl;

  NDArray::WaitAll();
	delete netExec;
  return TEST_SUCCESS;
}

void testNetReshape() {
	const std::vector<int> layers{100,100,1};
	const int max_epoch = 2;
	const float learning_rate = 0.1;

	auto net = test_mlp(layers);
	std::map<std::string, NDArray> netArgs;
  Context ctx = Context::cpu();
	//netArgs["X"] = NDArray({1,2,10}, Shape(1, 3), ctx);
	//netArgs["label"] = NDArray({1000},Shape(1), ctx);
	netArgs["X"] = NDArray({1,2,10,100,2,44,2,30,1}, Shape(3, 3), ctx);
	netArgs["label"] = NDArray({1000,900,800},Shape(3), ctx);
	net.InferArgsMap(ctx, &netArgs, netArgs);
  std::cout << "inputs: " << netArgs["X"] << std::endl;

  //TODO Batch forward method.
  //        Taking NDArray and shape s
  //        Slicing NDArray into s-shaped sub-arrays
  //        Running forward pass on sub-arrays
  //     Batch backward method.
  //        Can just be regular backward method for RL
  //          Works bc gradient comes from loss function
  //        Gradient may need to be expected value of gradient
  //          Gradient currently computes gradient wrt every input variable
  //          Each sample mean is a different variable
	auto initializer = Uniform(0.1);
	for (auto &arg : netArgs) {
    if (arg.first == "X" || arg.first == "label") continue;
		initializer(arg.first, &arg.second);
	}

	Optimizer *netOpt = OptimizerRegistry::Find("adam");
	netOpt->SetParam("lr", learning_rate);

	auto *netExec = net.SimpleBind(ctx, netArgs);
	auto netArgNames = net.ListArguments();

  for (int iter = 0; iter < max_epoch; ++iter) {
    // Compute gradients
    netExec->Forward(true);
    netExec->Backward();
    // Update parameters
    for (size_t i = 0; i < netArgNames.size(); ++i) {
      if (netArgNames[i] == "X" || netArgNames[i] == "label") continue;
      netOpt->Update(i, netExec->arg_arrays[i], netExec->grad_arrays[i]);
    }

    std::cout << "..." << std::endl;
  }

	netExec = net.SimpleBind(ctx, netArgs);
  netExec->Forward(false);
  std::cout << "input: " << netArgs["X"] << " output: " << netExec->outputs[0] << std::endl;

  netArgs["X"] = NDArray({1,2,10}, Shape(1, 3), ctx),
  netArgs["label"] = NDArray({1000}, Shape(1), ctx),
	net.InferArgsMap(ctx, &netArgs, netArgs);
	netExec = net.SimpleBind(ctx, netArgs);
  netExec->Forward(false);
  std::cout << "input: " << netArgs["X"] << " output: " << netExec->outputs[0] << std::endl;

  /*
  std::vector<NDArray> inVec = {
    NDArray({1,2,10}, Shape(1, 3), ctx),
    NDArray({2,34,100}, Shape(1, 3), ctx),
  };
  std::vector<NDArray> outVec;
  vecExecForward(
      netExec,
      net,
      netArgs,
      "X",
      inVec,
      outVec);
      */
  //std::cout << "output after reshape: " << netExec->outputs[0] << std::endl;
}

int testNDArrayBasics() {
  std::cout << "TEST: NDArray assignment" << std::endl;
  auto a1 = NDArray(Shape(2,2),ctx,true);
  a1 = 3.4;
  std::cout << "NDArray assignment: " << a1 << std::endl;
  std::cout << "------------------" << std::endl;
  std::cout << std::endl;

  NDArray::WaitAll();
  return TEST_SUCCESS;
}

int testSimpleArithmetic() {
  std::cout << "TEST: NDArray arithmetic" << std::endl;
  auto a1 = NDArray({2,3,4,3,2,4,3,4,2}, Shape(3,3), ctx);
  auto a2 = NDArray({10,10,10,20,20,20,30,30,30}, Shape(3,3), ctx);

  auto a3 = a1*a2;
  std::cout << "NDArray mult: " << a3 << std::endl;

  auto prod = dot(a1,a2);
  std::cout << "NDArray matmult: " << prod << std::endl;

  auto a4 = NDArray({1,1,1}, Shape(3,1), ctx);
  auto vecprod = dot(a1,a4);
  std::cout << "NDArray matrix - vector mult: "
            << a1 << "*" << a4 << "=" << vecprod << std::endl;

  std::cout << "------------------" << std::endl;
  std::cout << std::endl;

  NDArray::WaitAll();
  return TEST_SUCCESS;
}

int testNDArrayReduce() {
  std::cout << "TEST: NDArray reduce sum" << std::endl;
  auto a5 = NDArray({1,2,2,2,3,6,32,12,44,1,2,3,3,54,6,4},
                    Shape(2,2,4),
                    Context::cpu());
  auto a6 = sum(a5,1);
  std::cout << a5 << std::endl;
  std::cout << a6 << std::endl;

  auto a3 =  NDArray({1,2,2,2,3,6,32,12},
                    Shape(2,4),
                    Context::cpu());
  auto a4 = sum(a3,1).Reshape(Shape(2,1));
  std::cout << "a3: "
            << a3.GetShape()[0] << ","
            << a3.GetShape()[1] << std::endl;
  std::cout << "a4: " << a4 << std::endl
            << a4.GetShape()[0] << ","
            << a4.GetShape()[1] << std::endl;

  std::cout << "------------------" << std::endl;
  std::cout << std::endl;

  NDArray::WaitAll();
  return TEST_SUCCESS;
}

int testExp() {
  std::cout << "TEST: NDArray log" << std::endl;
  auto a1 = NDArray({1,2,2,2,3,6,32,12,44,1,2,3,3,54,6,4},
                    Shape(2,2,4),
                    Context::cpu());
  auto a2 = log(a1);
  std::cout << a2 << std::endl;
  std::cout << "------------------" << std::endl;
  std::cout << std::endl;

  NDArray::WaitAll();
  return TEST_SUCCESS;
}

int testNDArrayAssign() {
  std::cout << "TEST: NDArray assign" << std::endl;
  auto a1 = NDArray({1,2,2,2,3,6,32,12,44,1,2,3,3,54,6,4},
                    Shape(16,1),
                    Context::cpu());
  std::cout << "1d before: " << a1 << std::endl;

  a1.SetData(0,33);

  std::cout << "1d after: " << a1 << std::endl;

  auto a2 = a1.Reshape(Shape(4,4));

  std::cout << "2d before: " << a2 << std::endl;

  a2.SetData(1,1,65);

  std::cout << "2d after: " << a2 << std::endl;

  a1 = NDArray({1,2,4,5,55}, Shape(1,5), Context::cpu());
  std::cout << "Reassign: " << a1 << std::endl;

  a1 = NDArray({64,53,42,31},Shape(1,4),Context::cpu());
  a2.SetData(2,0,a1);

  std::cout << "Array insert: " << a2 << std::endl;

  std::cout << "------------------" << std::endl;
  std::cout << std::endl;

  NDArray::WaitAll();
  return TEST_SUCCESS;
}

int testNDArrayConcat() {
  std::cout << "TEST: NDArray concat" << std::endl;
  auto a1 = NDArray({1,2,2,2},
                    Shape(4,1),
                    Context::cpu());
  auto a2 = NDArray({3,6,32,12,44,1},
                    Shape(6,1),
                    Context::cpu());
  
  std::vector<NDArray> v = {a1,a2};
  auto a3 = Concat(v, Shape(10,1));
  std::cout << "concatenate: " << a1 << " + " << a2 << " = " << a3 << std::endl;

  auto a4 = Concat(a1,a2,Shape(10,1));
  std::cout << "concatenate2: " << a1 << " + " << a2 << " = " << a4 << std::endl;

  std::cout << "------------------" << std::endl;
  std::cout << std::endl;

  NDArray::WaitAll();
  return TEST_SUCCESS;
}

int testNDArrayInit() {
  std::cout << "TEST: NDArray initialize" << std::endl;

  auto a1 = zeros(Shape(3,4,2));
  std::cout << "zeros w/shape " << Shape(3,4,2) << ": " << a1 << std::endl;

  auto a2 = ones(Shape(3,4,2));
  std::cout << "ones w/shape " << Shape(3,4,2) << ": " << a2 << std::endl;

  std::cout << "------------------" << std::endl;
  std::cout << std::endl;

  NDArray::WaitAll();
  return TEST_SUCCESS;
}

int testNDArrayMax() {
  std::cout << "TEST: NDArray max" << std::endl;

  auto a1 = NDArray({3,6,32,12,44,1},
                    Shape(6,1),
                    Context::cpu());

  std::cout << "max of: " << a1 << ": " << max(a1) << std::endl;
  std::cout << "------------------" << std::endl;
  std::cout << std::endl;

  NDArray::WaitAll();
  return TEST_SUCCESS;
}

int testNDArrayCopy() {
  std::cout << "TEST: NDArray copy" << std::endl;

  auto a1 = NDArray({3,55,6,4,2},Shape(5,1),Context::cpu());
  auto a2 = a1.Copy(Context::cpu());

  std::cout << "a1: " << a1 << "\na2: " << a2 << std::endl;

  std::cout << "------------------" << std::endl;
  std::cout << std::endl;

  NDArray::WaitAll();
  return TEST_SUCCESS;
}

void testNDArrayIo() {
  auto a1 = NDArray({1,2,2,2},
                    Shape(4,1),
                    Context::cpu());
  auto a2 = NDArray({3,6,32,12,44,1},
                    Shape(6,1),
                    Context::cpu());

  std::string arrayvecfilename = "data/test/test_mxnet/arrayvecfile";
  std::vector<NDArray> ndarrayVec{a1,a2};
  NDArray::Save(arrayvecfilename,ndarrayVec);

  std::vector<NDArray> arrList;
  NDArray::Load(arrayvecfilename,&arrList);

  std::map<std::string,NDArray> ndarrayMap;
  ndarrayMap["a1"] = a1;
  ndarrayMap["a2"] = a2;
  std::string arraymapfilename = "data/test/test_mxnet/arraymapfile";
  NDArray::Save(arraymapfilename,ndarrayMap);

  std::map<std::string,NDArray> arrMap;
  NDArray::Load(arraymapfilename,nullptr,&arrMap);
  for (auto arr : arrMap) {
    std::cout << arr.first << ": " << arr.second << std::endl;
  }
}

int testSymbolReduce() {
  std::cout << "TEST: Symbol reduce" << std::endl;

  Symbol a = Symbol::Variable("a");
  Symbol b = Symbol::Variable("b");
  Symbol c = Symbol::Variable("c");

  //TODO see if sufficient to provide gradient of final output of multi-step
  //equation for network update
  std::map<std::string,NDArray> m; 
  m["a"] = NDArray({10,11,12,13},Shape(1,4),Context::cpu());
  m["b"] = NDArray({1,2,3,4},Shape(1,4),Context::cpu());
  m["c"] = NDArray({1000},Shape(1,1),Context::cpu());

  /*
  auto d = Reshape(mean(square(a+b),dmlc::optional<Shape>(Shape(1))),
                   Shape(m["c"].GetShape()))
           + c;
           */
  auto d = mean(square(a+b),dmlc::optional<Shape>(Shape(1)));
  auto e = reshape_like(d,c);
  auto f = e+c;

	f.InferArgsMap(Context::cpu(), &m, m);

  auto *exec = f.SimpleBind(Context::cpu(), m);

  exec->Forward(false);

  std::cout << "outputs shape: " << "(" << exec->outputs[0].GetShape()[0]
            << "," << exec->outputs[0].GetShape()[1] << ")" << std::endl;
  std::cout << "outputs: " << exec->outputs[0] << std::endl;
  std::cout << "------------------" << std::endl;
  std::cout << std::endl;

  NDArray::WaitAll();
  return TEST_SUCCESS;
}

int testSymbolIo() {
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

  }

	netArgs["X"] = NDArray({1,2,10,4,11,43,33,100,12}, Shape(3, 3), ctx);
	netArgs["label"] = NDArray({1000,900,700},Shape(3), ctx);
	net.InferArgsMap(ctx, &netArgs, netArgs);
	netExec = net.SimpleBind(ctx, netArgs);
  netExec->Forward(false);
  std::cout << "output: " << netExec->outputs[0] << std::endl;

  std::string netfilename = "data/test/test_mxnet/sampleNet.txt";
  net.Save(netfilename);

  auto netCpy = Symbol::Load(netfilename);
	netCpy.InferArgsMap(ctx, &netArgs, netArgs);
	auto *netCpyExec = netCpy.SimpleBind(ctx, netArgs);
  netCpyExec->Forward(false);
  std::cout << "output: " << netCpyExec->outputs[0] << std::endl;

  std::cout << "------------------" << std::endl;
  std::cout << std::endl;

  NDArray::WaitAll();
	delete netExec;
	delete netCpyExec;
  return TEST_SUCCESS;

}

int testLogging() {
  std::cout << "TEST: logging" << std::endl;

  Symbol a = Symbol::Variable("a");
  Symbol b = Symbol::Variable("b");
  Symbol c = Symbol::Variable("c");

  //TODO see if sufficient to provide gradient of final output of multi-step
  //equation for network update
  std::map<std::string,NDArray> m; 
  m["a"] = NDArray({10,11,12,13},Shape(1,4),Context::cpu());
  m["b"] = NDArray({1,2,3,4},Shape(1,4),Context::cpu());
  m["c"] = NDArray({1000},Shape(1,1),Context::cpu());

  /*
  auto d = Reshape(mean(square(a+b),dmlc::optional<Shape>(Shape(1))),
                   Shape(m["c"].GetShape()))
           + c;
           */
  auto d = mean(square(a+b),dmlc::optional<Shape>(Shape(1)));
  auto e = reshape_like(d,c);
  auto f = e+c;

	f.InferArgsMap(Context::cpu(), &m, m);

  auto *exec = f.SimpleBind(Context::cpu(), m);

  exec->Forward(false);

  
  std::string datafilename = "data/test/test_mxnet/tmp.txt";
  remove(datafilename.c_str());
  std::ofstream logfile;
  logfile.open(datafilename);

  std::cout << "writing to " << datafilename << std::endl;

  logfile << m["a"] << std::endl;
  logfile << m["b"] << std::endl;
  logfile << exec->outputs[0] << std::endl;;

  logfile.close();
  std::cout << "------------------" << std::endl;
  std::cout << std::endl;

  NDArray::WaitAll();
  return TEST_SUCCESS;

}

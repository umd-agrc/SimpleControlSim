#include "utils.h"

using namespace mxnet::cpp;

static Context ctx = Context::cpu();

void vecExecForward(
    mxnet::cpp::Executor *exec, 
    mxnet::cpp::Symbol &sym,
    std::map<std::string,mxnet::cpp::NDArray> &inMap,
    std::string arrName,
    const std::vector<mxnet::cpp::NDArray> &inVec,
    std::vector<mxnet::cpp::NDArray> &outVec) {
//net.InferArgsMap(ctx, &netArgs, netArgs);
  for (auto arr : inVec) {
    inMap[arrName] = arr;
    sym.InferArgsMap(ctx,&inMap,inMap);
    exec = sym.SimpleBind(ctx, inMap);
    exec->Forward(false);
    outVec.push_back(exec->outputs[0]);
    std::cout << "input: " << arr << " gettin outputs: " << exec->outputs[0] << std::endl;
  }
}

void vecExecForwardBackward(
    mxnet::cpp::Executor *exec, 
    mxnet::cpp::Symbol &sym,
    std::map<std::string,mxnet::cpp::NDArray> &inMap,
    std::vector<mxnet::cpp::NDArray> &labelVec,
    std::string arrName,
    const std::vector<mxnet::cpp::NDArray> &inVec,
    std::vector<mxnet::cpp::NDArray> &outVec) {

}

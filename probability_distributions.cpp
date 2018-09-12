#include "probability_distributions.h"

using namespace mxnet::cpp;

DiagGaussianPd::DiagGaussianPd() {
}

NDArray DiagGaussianPd::neglogp(
    std::map<std::string,NDArray> &trajSegment) {
  return (sum(square((trajSegment["action"]-trajSegment["mean"])
                     /trajSegment["std"]),1)*0.5
          + 0.5*log(2*M_PI)*NUM_INPUTS
          + sum(trajSegment["logstd"],1)).Reshape(
                Shape(trajSegment["action"].GetShape()[0],1));
}

NDArray DiagGaussianPd::logp(
    std::map<std::string,NDArray> &trajSegment) {
  return negative(neglogp(trajSegment));
}

NDArray DiagGaussianPd::kl(
  std::map<std::string,NDArray> &trajSegment,
  std::map<std::string,NDArray> &oldTrajSegment) {
  return sum(oldTrajSegment["logstd"] -
             trajSegment["logstd"]
             + (square(trajSegment["std"])
                + square(trajSegment["mean"] - oldTrajSegment["mean"]))
             / (square(oldTrajSegment["std"])*2) - 0.5,1).Reshape(
                    Shape(trajSegment["logstd"].GetShape()[0],1));
}

NDArray DiagGaussianPd::entropy(
    std::map<std::string,NDArray> &trajSegment) {
  return sum(trajSegment["logstd"]+0.5*log(2*M_PI*M_E),1).Reshape(
            Shape(trajSegment["logstd"].GetShape()[0],1));
}

Symbol DiagGaussianPd::neglogp(Symbol &meanCtl,
               Symbol &std,
               Symbol &logstd,
               Symbol &action) {
  auto a = sum(square((action-meanCtl)/std),
               dmlc::optional<Shape>(Shape(1)))*0.5
           + 0.5*log(2*M_PI)*NUM_INPUTS;
  auto b = sum(logstd,dmlc::optional<Shape>(Shape(1)));
  auto c = reshape_like(a,b);
  return c + b;
}

Symbol DiagGaussianPd::logp(Symbol &meanCtl,
            Symbol &std,
            Symbol &logstd,
            Symbol &action) {
  return negative(neglogp(meanCtl,std,logstd,action));
}

Symbol DiagGaussianPd::kl(Symbol &meanCtl,
          Symbol &std,
          Symbol &logstd,
          Symbol &oldMeanCtl,
          Symbol &oldStd,
          Symbol &oldLogstd) {
  return sum(oldLogstd
             - logstd
             + (square(std)
                + square(meanCtl - oldMeanCtl))
             / (square(oldStd)*2) - 0.5,dmlc::optional<Shape>(Shape(1)));
}

Symbol DiagGaussianPd::entropy(Symbol &logstd) {
  return sum(logstd+0.5*log(2*M_PI*M_E),dmlc::optional<Shape>(Shape(1)));
}


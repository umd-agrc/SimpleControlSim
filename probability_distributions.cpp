#include "probability_distributions.h"

using namespace mxnet::cpp;

DiagGaussianPd::DiagGaussianPd() {
}

NDArray DiagGaussianPd::neglogp(
    std::map<std::string,NDArray> &trajSegment) {
   //TODO make sure summing over correct axes
   return sum(square((trajSegment["action"]-trajSegment["mean"])
                         /trajSegment["std"]))*0.5
          + 0.5*log(2*M_PI)*NUM_INPUTS
          + sum(trajSegment["logstd"]);
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
             / (square(oldTrajSegment["std"])*2) - 0.5);
}

NDArray DiagGaussianPd::entropy(
    std::map<std::string,NDArray> &trajSegment) {
  return sum(trajSegment["logstd"]+0.5*log(2*M_PI*M_E));
}



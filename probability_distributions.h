#pragma once

#include <cmath>
#include <cstring>
#include <string>

#include "defines.h"
#include "utils.h"

//TODO overload equality operator so that pd can be redefined after update
class DiagGaussianPd {
public:
  int len_;

public:
  DiagGaussianPd();
  
  mxnet::cpp::NDArray neglogp(
      std::map<std::string,mxnet::cpp::NDArray> &trajSegment);

  mxnet::cpp::NDArray logp(
      std::map<std::string,mxnet::cpp::NDArray> &trajSegment);

  mxnet::cpp::NDArray kl(
    std::map<std::string,mxnet::cpp::NDArray> &trajSegment,
    std::map<std::string,mxnet::cpp::NDArray> &oldTrajSegment);

  mxnet::cpp::NDArray entropy(
    std::map<std::string,mxnet::cpp::NDArray> &trajSegment);

  mxnet::cpp::Symbol neglogp(mxnet::cpp::Symbol &mean,
                             mxnet::cpp::Symbol &std,
                             mxnet::cpp::Symbol &logstd,
                             mxnet::cpp::Symbol &action);

  mxnet::cpp::Symbol logp(mxnet::cpp::Symbol &mean,
                          mxnet::cpp::Symbol &std,
                          mxnet::cpp::Symbol &logstd,
                          mxnet::cpp::Symbol &action);

  mxnet::cpp::Symbol kl(mxnet::cpp::Symbol &mean,
                        mxnet::cpp::Symbol &std,
                        mxnet::cpp::Symbol &logstd,
                        mxnet::cpp::Symbol &oldMean,
                        mxnet::cpp::Symbol &oldStd,
                        mxnet::cpp::Symbol &oldLogstd);

  mxnet::cpp::Symbol entropy(mxnet::cpp::Symbol &logstd);

};



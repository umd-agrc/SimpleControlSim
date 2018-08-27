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
};



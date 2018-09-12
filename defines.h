#pragma once

#include <stdio.h>
#include <stdbool.h>
#include <vector>
#include <sys/stat.h>
#include <unistd.h>
#include <string>
#include <iostream>
#include <utility>

#include <mxnet-cpp/MxNetCpp.h>

extern "C" {
#include <cblas.h>
}

#define SIM_SUCCESS 0
#define SIM_FAIL -1

#define TEST_SUCCESS 0
#define TEST_FAIL -1

#define SHUTDOWN_MESSAGE "shutdown\n"

#define LQR_TYPE 0
#define POLICY_TYPE 1
#define CAMILA_TYPE 2

#define NUM_STATES 12
#define NUM_INPUTS 4

struct VehicleState {
  mxnet::cpp::NDArray yd,y;
};

struct Controller {
  mxnet::cpp::NDArray *(*feedback) (
      mxnet::cpp::NDArray &yd,
      mxnet::cpp::NDArray &y,
      mxnet::cpp::NDArray *baseAction,
      mxnet::cpp::NDArray *meanAction);
};

struct DataPoint {
  mxnet::cpp::NDArray inTarget;
  mxnet::cpp::NDArray in;
  mxnet::cpp::NDArray out;
  mxnet::cpp::NDArray advantageValues;
  mxnet::cpp::NDArray value;
  mxnet::cpp::NDArray valueTarget;
  double reward;
  double objectiveValue;
  mxnet::cpp::NDArray gradientValues;

  DataPoint() {}
  DataPoint(std::vector<mx_float> inVec, std::vector<mx_float> outVec) {
    in = mxnet::cpp::NDArray(inVec,
                             mxnet::cpp::Shape(inVec.size(), 1),
                             mxnet::cpp::Context::cpu());
    out = mxnet::cpp::NDArray(outVec,
                              mxnet::cpp::Shape(outVec.size(), 1),
                              mxnet::cpp::Context::cpu());
  }
  DataPoint(mxnet::cpp::NDArray inVec, mxnet::cpp::NDArray outVec) : 
  in(inVec), out(outVec) {}
  DataPoint(mxnet::cpp::NDArray inVec, mxnet::cpp::NDArray outVec,
      mxnet::cpp::NDArray adv, mxnet::cpp::NDArray obj) :
    in(inVec), out(outVec), advantageValues(adv) {
    objectiveValue = obj.At(0,0);  
  }
};

/*
struct PolicyFunction {
  mxnet::cpp::Symbol policyNet;
  mxnet::cpp::Symbol oldPolicyNet;
  mxnet::cpp::Symbol valueNet;
  DiagGaussianDistribution probabilityDistribution;
};
*/

struct LossAndGradients {
  double loss;
  mxnet::cpp::NDArray gradients;
};

// Check if file exists
inline bool fileExists(const std::string& name) {
  struct stat buffer;
  return (stat (name.c_str(), &buffer) == 0);
}

inline void log(){std::cout << std::endl;}

template<typename First, typename ...Rest>
inline void log(First &&first, Rest && ...rest) {
  std::cout << std::forward<First> (first);
  log(std::forward<Rest>(rest)...);
}

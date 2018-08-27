#pragma once

#include <stdio.h>
#include <stdbool.h>
//#include <gsl/gsl_matrix.h>
//#include <gsl/gsl_vector.h>
#include <vector>
#include <sys/stat.h>
#include <unistd.h>
#include <string>

#include <mxnet-cpp/MxNetCpp.h>

extern "C" {
#include <cblas.h>
}

#define SIM_SUCCESS 0
#define SIM_FAILURE -1

#define SIM_INFO(...) do {fprintf(stdout,##__VA_ARGS__);} while(0)
#define SIM_ERROR(...) do {fprintf(stdout,##__VA_ARGS__);} while(0)

#define SHUTDOWN_MESSAGE "shutdown\n"

#define LQR_TYPE 0
#define POLICY_TYPE 1
#define CAMILA_TYPE 2

#define NUM_STATES 12
#define NUM_INPUTS 4

struct VehicleState {
  std::vector<mx_float> yd;
  std::vector<mx_float> y;
};

struct Controller {
  std::vector<mx_float> *(*feedback) (
      const std::vector <mx_float> *yd,
      const std::vector<mx_float> *y,
      std::vector<mx_float> *baseAction,
      std::vector<mx_float> *meanAction);
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

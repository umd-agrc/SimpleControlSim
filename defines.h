#pragma once

#include <stdio.h>
#include <stdbool.h>
//#include <gsl/gsl_matrix.h>
//#include <gsl/gsl_vector.h>
#include <vector>
#include <sys/stat.h>
#include <unistd.h>
#include <string>

#include <tiny_dnn/tiny_dnn.h>

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

struct VehicleState {
  std::vector<double> yd;
  std::vector<double> y;
  //gsl_vector *yd;
  //gsl_vector *y;
};

struct Controller {
  std::vector<double> *(*feedback) (const std::vector <double> *yd,
                                    const std::vector<double> *y);
};

struct DataPoint {
  tiny_dnn::vec_t inTarget;
  tiny_dnn::vec_t in;
  tiny_dnn::vec_t out;
  tiny_dnn::vec_t advantageValues;
  tiny_dnn::vec_t value;
  tiny_dnn::vec_t valueTarget;
  tiny_dnn::float_t reward;
  tiny_dnn::float_t objectiveValue;
  tiny_dnn::vec_t gradientValues;

  DataPoint() {}
  DataPoint(std::vector<double> inVec, std::vector<double> outVec) {
    size_t k = inVec.size();
    size_t l = outVec.size();

    in.resize(k);
    out.resize(l);

    for (size_t i = 0; i < k; i++) {
      in[i] = inVec[i];
    }

    for (size_t i = 0; i < l; i++) {
      out[i] = outVec[i];
    }
  }
  DataPoint(tiny_dnn::vec_t inVec, tiny_dnn::vec_t outVec) : 
  in(inVec), out(outVec) {}
  DataPoint(tiny_dnn::vec_t inVec, tiny_dnn::vec_t outVec,
      tiny_dnn::vec_t adv, tiny_dnn::float_t obj) :
    in(inVec), out(outVec), advantageValues(adv), objectiveValue(obj) {}
};

struct PolicyFunction {
  tiny_dnn::network<tiny_dnn::graph> policyNet;
  tiny_dnn::network<tiny_dnn::graph> oldPolicyNet;
  tiny_dnn::network<tiny_dnn::graph> valueNet;
  tiny_dnn::diag_gaussian_distribution probabilityDistribution;
};

struct LossAndGradients {
  tiny_dnn::float_t loss;
  tiny_dnn::vec_t gradients;
};

// Check if file exists
inline bool fileExists(const std::string& name) {
  struct stat buffer;
  return (stat (name.c_str(), &buffer) == 0);
}
